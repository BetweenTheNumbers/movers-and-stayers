"""
List the players affected by unresolved geocodes.

Cross-references the geocode cache against the analysis dataset and reports
every draftee whose birth city or high-school city still has no coordinates,
with name, draft year, team, and round.

Usage:
    python scripts/unresolved_players.py            # print report
    python scripts/unresolved_players.py --csv      # also write a CSV

Inputs (uses whichever exist):
    geocode_cache_v3.json
    v3_analysis.csv          preferred (has parsed city/state + names)
    data/tbc_draft_register.csv   fallback

Output:
    unresolved_players.csv   (with --csv)
"""

import json
import os
import sys
import pandas as pd

CACHE_FILE = 'geocode_cache_v3.json'
ANALYSIS = 'v3_analysis.csv'
REGISTER = 'data/tbc_draft_register.csv'
OUT_CSV = 'unresolved_players.csv'


def load_unresolved_keys():
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run from the project root.")
        sys.exit(1)
    with open(CACHE_FILE, encoding='utf-8') as f:
        cache = json.load(f)
    bad = {k for k, v in cache.items()
           if v.get('lat') is None or v.get('lon') is None}
    return bad, len(cache)


def norm_key(city, state):
    if pd.isna(city) or pd.isna(state):
        return None
    c, s = str(city).strip(), str(state).strip()
    if not c or not s:
        return None
    return f"{c}|{s}"


def pick(df, *names):
    """First column present from names."""
    for n in names:
        if n in df.columns:
            return n
    return None


def main():
    bad_keys, total = load_unresolved_keys()
    print(f"\nGeocode cache: {total:,} entries, {len(bad_keys)} unresolved")
    if not bad_keys:
        print("Nothing unresolved. All locations have coordinates.\n")
        return

    src = ANALYSIS if os.path.exists(ANALYSIS) else REGISTER
    if not os.path.exists(src):
        print(f"ERROR: neither {ANALYSIS} nor {REGISTER} found.")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    print(f"Scanning {src} ({len(df):,} rows)\n")

    bcity = pick(df, 'birth_city')
    bstate = pick(df, 'birth_state')
    hcity = pick(df, 'hs_city', 'school_city')
    hstate = pick(df, 'hs_state', 'school_state')

    # If reading the raw register, parse place/school on the fly
    if bcity is None and 'place' in df.columns:
        parsed = df['place'].astype(str).str.rsplit(',', n=1, expand=True)
        df['birth_city'] = parsed[0].str.strip()
        df['birth_state'] = parsed[1].str.strip() if parsed.shape[1] > 1 else None
        bcity, bstate = 'birth_city', 'birth_state'
    if hcity is None and 'school' in df.columns:
        ex = df['school'].astype(str).str.extract(r'\(([^()]+)\)\s*$')[0]
        parsed = ex.str.rsplit(',', n=1, expand=True)
        df['hs_city'] = parsed[0].str.strip() if parsed is not None else None
        df['hs_state'] = parsed[1].str.strip() if parsed.shape[1] > 1 else None
        hcity, hstate = 'hs_city', 'hs_state'

    fn = pick(df, 'firstName', 'first_name')
    ln = pick(df, 'lastName', 'last_name')
    team = pick(df, 'Teamname', 'team', 'Team')
    rnd = pick(df, 'draftRound', 'round')
    yr = pick(df, 'year')
    cls = pick(df, 'playerClass')
    lvl = pick(df, 'highLevel')

    rows = []
    for _, r in df.iterrows():
        bk = norm_key(r.get(bcity), r.get(bstate)) if bcity else None
        hk = norm_key(r.get(hcity), r.get(hstate)) if hcity else None
        missing = []
        if bk in bad_keys:
            missing.append(f"birth={bk.replace('|', ', ')}")
        if hk in bad_keys:
            missing.append(f"school={hk.replace('|', ', ')}")
        if not missing:
            continue
        name = f"{r.get(fn, '')} {r.get(ln, '')}".strip() if fn and ln else '(name n/a)'
        rows.append({
            'player': name,
            'year': r.get(yr, ''),
            'team': r.get(team, ''),
            'round': r.get(rnd, ''),
            'class': r.get(cls, ''),
            'reached': r.get(lvl, ''),
            'unresolved': '; '.join(missing),
        })

    if not rows:
        print("No players in the analysis file use the unresolved locations.")
        print("(They may sit outside the draft-year window.)\n")
        return

    out = pd.DataFrame(rows).sort_values(['unresolved', 'year'])
    print(f"{len(out)} affected draft rows:\n")
    print(out.to_string(index=False))

    # summary by location
    print(f"\nBy unresolved location:")
    print(out['unresolved'].value_counts().to_string())

    if '--csv' in sys.argv:
        out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {OUT_CSV}")
    print()


if __name__ == '__main__':
    main()
