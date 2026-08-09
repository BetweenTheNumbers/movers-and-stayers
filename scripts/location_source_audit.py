"""
Location-source audit (utility, not a pipeline step).

Before committing to a long geocoding run, this reports exactly how many
players have a usable HS location, from which source, by era -- and how many
NEW (city, state) pairs would need geocoding.

Where HS locations come from:
  - HS draftees:      the register's `school` field, "Name (City,ST)"
  - College draftees: `hsplace` in the signing-bonus file (SIGNED players only;
                      the register's `school` holds a college name for them)

Usage:
    python scripts/location_source_audit.py
    python scripts/location_source_audit.py --from 1965 --to 2025
    python scripts/location_source_audit.py --csv

Output:
    location_source_audit.csv        (with --csv)
    geocode_todo_preview.csv         unique places not yet cached
"""

import json
import os
import re
import sys
import pandas as pd

REGISTER = 'data/tbc_draft_register.csv'
BONUS = 'data/tbc_signing_bonus.csv'
CACHE_FILE = 'geocode_cache_v3.json'
OUT_CSV = 'location_source_audit.csv'
TODO_CSV = 'geocode_todo_preview.csv'

ERAS = [(1965, 1979), (1980, 1986), (1987, 1995), (1996, 2005),
        (2006, 2019), (2020, 2025)]


def arg(flag, default):
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def parse_place(p):
    if pd.isna(p) or ',' not in str(p):
        return None, None
    a, b = str(p).rsplit(',', 1)
    return a.strip(), b.strip()


def parse_school_city(s):
    if pd.isna(s):
        return None, None
    m = re.search(r'\(([^()]+)\)\s*$', str(s))
    if not m or ',' not in m.group(1):
        return None, None
    a, b = m.group(1).rsplit(',', 1)
    return a.strip(), b.strip()


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def main():
    if not os.path.exists(REGISTER):
        print(f"ERROR: {REGISTER} not found. Run from the project root.")
        sys.exit(1)

    y0, y1 = arg('--from', 1965), arg('--to', 2025)
    reg = pd.read_csv(REGISTER, low_memory=False)
    reg = reg[(reg['year'] >= y0) & (reg['year'] <= y1)].copy()
    print(f"\nRegister rows {y0}-{y1}: {len(reg):,}")

    # birth
    bc = reg['place'].apply(lambda p: pd.Series(parse_place(p)))
    reg['birth_city'], reg['birth_state'] = bc[0], bc[1]
    # HS from register school field
    sc = reg['school'].apply(lambda s: pd.Series(parse_school_city(s)))
    reg['reg_hs_city'], reg['reg_hs_state'] = sc[0], sc[1]

    reg['is_hs'] = ((reg.get('playerClass') == 'HS') |
                    (reg.get('schoolDivision') == 'HS'))

    # HS from signing-bonus hsplace
    reg_pid = pick(reg, 'PlayerID', 'playerid')
    bonus_map = {}
    if os.path.exists(BONUS):
        sb = pd.read_csv(BONUS, low_memory=False)
        sb_pid = pick(sb, 'playerid', 'PlayerID')
        hsp = pick(sb, 'hsplace')
        print(f"Signing-bonus rows: {len(sb):,}"
              + (f"; hsplace present for {sb[hsp].notna().sum():,}" if hsp else ""))
        if sb_pid and hsp:
            t = sb[[sb_pid, hsp]].dropna()
            bonus_map = dict(zip(t[sb_pid].astype(str).str.strip(),
                                 t[hsp].astype(str).str.strip()))
    else:
        print(f"(no {BONUS} found)")

    if reg_pid:
        keys = reg[reg_pid].astype(str).str.strip()
        bp = keys.map(bonus_map)
        parsed = bp.apply(lambda p: pd.Series(parse_place(p)))
        reg['sb_hs_city'], reg['sb_hs_state'] = parsed[0], parsed[1]
    else:
        reg['sb_hs_city'] = reg['sb_hs_state'] = None

    # unified HS location: register first for HS draftees, else signing bonus
    reg['hs_city'] = reg['reg_hs_city'].fillna(reg['sb_hs_city'])
    reg['hs_state'] = reg['reg_hs_state'].fillna(reg['sb_hs_state'])
    reg['hs_source'] = 'none'
    reg.loc[reg['sb_hs_city'].notna(), 'hs_source'] = 'signing_bonus'
    reg.loc[reg['reg_hs_city'].notna(), 'hs_source'] = 'register_school'

    reg['has_birth'] = reg['birth_city'].notna() & reg['birth_state'].notna()
    reg['has_hs'] = reg['hs_city'].notna() & reg['hs_state'].notna()
    reg['usable'] = reg['has_birth'] & reg['has_hs']

    print(f"\n{'=' * 92}")
    print("USABLE ROWS BY ERA (birth AND high-school location both present)")
    print("=" * 92)
    print(f"{'Era':<12} {'Rows':>8} {'Birth':>8} {'HS any':>8} {'Usable':>8} "
          f"{'from reg':>10} {'from bonus':>11} {'usable%':>8}")
    print("-" * 92)
    rows = []
    for lo, hi in ERAS:
        g = reg[(reg['year'] >= lo) & (reg['year'] <= hi)]
        if g.empty:
            continue
        u = g[g['usable']]
        r_ = int((u['hs_source'] == 'register_school').sum())
        b_ = int((u['hs_source'] == 'signing_bonus').sum())
        print(f"{f'{lo}-{hi}':<12} {len(g):>8,} {int(g['has_birth'].sum()):>8,} "
              f"{int(g['has_hs'].sum()):>8,} {len(u):>8,} {r_:>10,} {b_:>11,} "
              f"{len(u)/len(g)*100:>7.1f}%")
        rows.append({'era': f'{lo}-{hi}', 'rows': len(g),
                     'has_birth': int(g['has_birth'].sum()),
                     'has_hs': int(g['has_hs'].sum()), 'usable': len(u),
                     'from_register': r_, 'from_signing_bonus': b_,
                     'usable_pct': round(len(u)/len(g)*100, 1)})

    # HS vs college split
    print(f"\n{'=' * 92}")
    print("USABLE ROWS BY ERA AND PATH")
    print("=" * 92)
    print(f"{'Era':<12} {'HS draftees':>13} {'HS usable':>11} {'College':>10} "
          f"{'Coll usable':>13} {'Coll cover%':>12}")
    print("-" * 92)
    for lo, hi in ERAS:
        g = reg[(reg['year'] >= lo) & (reg['year'] <= hi)]
        if g.empty:
            continue
        hs, col = g[g['is_hs']], g[~g['is_hs']]
        cu = int(col['usable'].sum())
        print(f"{f'{lo}-{hi}':<12} {len(hs):>13,} {int(hs['usable'].sum()):>11,} "
              f"{len(col):>10,} {cu:>13,} "
              f"{(cu/len(col)*100 if len(col) else 0):>11.1f}%")

    # geocoding cost
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding='utf-8') as f:
            cache = json.load(f)
    have = {k for k, v in cache.items()
            if v.get('lat') is not None and v.get('lon') is not None}

    needed = set()
    for _, r in reg[reg['has_birth']].iterrows():
        needed.add((str(r['birth_city']).strip(), str(r['birth_state']).strip()))
    for _, r in reg[reg['has_hs']].iterrows():
        needed.add((str(r['hs_city']).strip(), str(r['hs_state']).strip()))
    todo = sorted(p for p in needed if f"{p[0]}|{p[1]}" not in have)

    print(f"\n{'=' * 92}")
    print("GEOCODING COST")
    print("=" * 92)
    print(f"  Unique places required:  {len(needed):,}")
    print(f"  Already cached:          {len(needed) - len(todo):,}")
    print(f"  NEW places to geocode:   {len(todo):,}")
    mins = len(todo) / 60.0
    print(f"  Estimated time at 1/sec: {mins:.0f} min ({mins/60:.1f} hours)")

    if todo:
        pd.DataFrame(todo, columns=['city', 'state']).to_csv(
            TODO_CSV, index=False, encoding='utf-8-sig')
        print(f"  Preview written to {TODO_CSV}")
        st = pd.Series([s for _, s in todo]).value_counts()
        print(f"\n  Top states/countries among new places:")
        for k, v in st.head(12).items():
            print(f"    {k}: {v:,}")

    if '--csv' in sys.argv and rows:
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {OUT_CSV}")
    print()


if __name__ == '__main__':
    main()
