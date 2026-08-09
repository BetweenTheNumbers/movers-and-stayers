"""
Find and categorize HS draftees with no computed distance.

Distance requires BOTH a birth location and a high-school location, each of
which must (a) be present in the source data and (b) resolve to coordinates.
This script separates those failure modes so you can describe them precisely.

Usage:
    python scripts/missing_distance_players.py           # report
    python scripts/missing_distance_players.py --csv     # also write CSV
    python scripts/missing_distance_players.py --all     # include non-HS draftees

Output:
    missing_distance_players.csv   (with --csv)
"""

import json
import os
import sys
import pandas as pd

ANALYSIS = 'v3_analysis.csv'
CACHE_FILE = 'geocode_cache_v3.json'
OUT_CSV = 'missing_distance_players.csv'


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def blank(v):
    return pd.isna(v) or str(v).strip() in ('', '--', 'nan', 'None')


def main():
    if not os.path.exists(ANALYSIS):
        print(f"ERROR: {ANALYSIS} not found. Run from the project root.")
        sys.exit(1)
    df = pd.read_csv(ANALYSIS, low_memory=False)

    good_keys = set()
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding='utf-8') as f:
            cache = json.load(f)
        good_keys = {k for k, v in cache.items()
                     if v.get('lat') is not None and v.get('lon') is not None}

    bcity, bstate = pick(df, 'birth_city'), pick(df, 'birth_state')
    hcity = pick(df, 'hs_city', 'school_city')
    hstate = pick(df, 'hs_state', 'school_state')
    fn, ln = pick(df, 'firstName'), pick(df, 'lastName')
    yr, team = pick(df, 'year'), pick(df, 'Teamname', 'team')
    rnd, cls = pick(df, 'draftRound'), pick(df, 'playerClass')
    lvl = pick(df, 'highLevel')

    hs_only = '--all' not in sys.argv
    sub = df
    if hs_only and 'is_hs_draftee' in df.columns:
        sub = df[df['is_hs_draftee'] == 1]
    missing = sub[sub['distance_miles'].isna()].copy()

    label = 'HS draftees' if hs_only else 'all draftees'
    print(f"\n{label}: {len(sub):,} rows, {len(missing):,} without a distance "
          f"({len(missing)/max(len(sub),1)*100:.1f}%)\n")
    if missing.empty:
        print("Nothing missing.\n")
        return

    rows = []
    for _, r in missing.iterrows():
        b_present = not (blank(r.get(bcity)) or blank(r.get(bstate)))
        h_present = not (blank(r.get(hcity)) or blank(r.get(hstate)))
        b_key = f"{str(r.get(bcity)).strip()}|{str(r.get(bstate)).strip()}" if b_present else None
        h_key = f"{str(r.get(hcity)).strip()}|{str(r.get(hstate)).strip()}" if h_present else None
        b_geo = (b_key in good_keys) if b_present else False
        h_geo = (h_key in good_keys) if h_present else False

        if not b_present and not h_present:
            reason = 'both locations absent in source'
        elif not b_present:
            reason = 'birth location absent in source'
        elif not h_present:
            reason = 'HS location absent in source'
        elif not b_geo and not h_geo:
            reason = 'both locations present but ungeocoded'
        elif not b_geo:
            reason = 'birth location present but ungeocoded'
        elif not h_geo:
            reason = 'HS location present but ungeocoded'
        else:
            reason = 'both geocoded — check compute step'

        rows.append({
            'player': f"{r.get(fn, '')} {r.get(ln, '')}".strip(),
            'year': r.get(yr, ''), 'team': r.get(team, ''),
            'round': r.get(rnd, ''), 'class': r.get(cls, ''),
            'reached': r.get(lvl, ''),
            'birth': '' if not b_present else f"{r.get(bcity)}, {r.get(bstate)}",
            'hs': '' if not h_present else f"{r.get(hcity)}, {r.get(hstate)}",
            'reason': reason,
        })

    out = pd.DataFrame(rows)
    print("Reason breakdown:")
    print(out['reason'].value_counts().to_string())

    print(f"\nReached MLB among these: "
          f"{(out['reached'] == 'MLB').sum()} of {len(out)}")

    print("\nFirst 30 rows:")
    print(out.head(30).to_string(index=False))

    # Always export. If the file is open in Excel, fall back to a timestamped name.
    try:
        out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {OUT_CSV} ({len(out)} rows, {len(out.columns)} columns)")
    except (PermissionError, OSError):
        import time
        alt = OUT_CSV.replace('.csv', f"_{time.strftime('%H%M%S')}.csv")
        out.to_csv(alt, index=False, encoding='utf-8-sig')
        print(f"\n{OUT_CSV} was locked (open in Excel?); saved {alt} instead.")
    print()


if __name__ == '__main__':
    main()
