"""
Data-integrity check: for the ~1,681 college players who have HS location from
BOTH routes (own HS-draft row AND signing-bonus hsplace), do the two sources
AGREE? If they agree, the union is trustworthy. If they conflict often, one
source is unreliable and we must resolve which before building.

Compares at two levels:
  - STATE match (lenient): same state?
  - CITY match (strict):   same city (normalized)?
Prints conflict examples so we can eyeball which source is wrong.

Read-only.

Run:  python scripts/diag_hs_conflict.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def parse_hsplace(p):
    p = str(p).strip()
    if ',' in p and p not in ('--,--', 'nan'):
        c, s = p.rsplit(',', 1)
        return c.strip(), s.strip()
    return None


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def norm_city(c):
    c = str(c).lower().strip()
    c = re.sub(r'[.\'`]', '', c)
    c = re.sub(r'\s+', ' ', c)
    c = c.replace('saint ', 'st ').replace('ft ', 'fort ')
    return c


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'

    # backlink HS by PlayerID
    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    hs_rows = hs_rows.loc[hs_rows['loc'].notna()]
    backlink = {}
    for _, r in hs_rows.iterrows():
        if r['PlayerID'] not in backlink:
            backlink[r['PlayerID']] = r['loc']

    # hsplace by PlayerID
    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsplace = find(sb, 'hsplace')
    hsp = {}
    for _, r in sb.iterrows():
        loc = parse_hsplace(r[hsplace])
        if loc and r[sb_pid] not in hsp:
            hsp[r[sb_pid]] = loc

    # players with BOTH
    both_ids = set(backlink) & set(hsp)
    print(f"Players with HS location from BOTH routes: {len(both_ids):,}\n")

    state_match = city_match = 0
    conflicts = []
    for pid in both_ids:
        bc, bs = backlink[pid]
        hc, hs = hsp[pid]
        sm = bs.upper() == hs.upper()
        cm = norm_city(bc) == norm_city(hc)
        state_match += sm
        city_match += cm
        if not sm:  # state-level conflict is the serious one
            conflicts.append((pid, f"{bc},{bs}", f"{hc},{hs}"))

    n = len(both_ids)
    print("="*60)
    print("AGREEMENT between backlink and hsplace")
    print("="*60)
    print(f"  same STATE: {state_match:,}/{n:,} ({state_match/n*100:.1f}%)")
    print(f"  same CITY:  {city_match:,}/{n:,} ({city_match/n*100:.1f}%)")
    print(f"  state-level CONFLICTS: {len(conflicts):,}")

    if conflicts:
        print("\n  Sample state conflicts (backlink  vs  hsplace):")
        for pid, b, h in conflicts[:20]:
            print(f"    PlayerID {pid}:  {b:<22} vs  {h}")

    print("\n  READ: high city-match => sources agree, union is safe. Many")
    print("  state conflicts => one source is unreliable; inspect before use.")


if __name__ == '__main__':
    main()
