"""
Definitive HS-location coverage for COLLEGE draftees (1996-2019). Combines every
route we validated and reports the final gap -- what % simply cannot get a HS
location by any method.

Routes to a college player's HS city:
  A. BACKLINK  - the player's own earlier HS-draft register row (any year),
                 HS parsed from the 'school' field '(City,ST)'.
  B. HSPLACE   - the hsplace field in tbc_signing_bonus.csv.
  (birth city is on the college row already via 'place', ~100%.)

Reports: A alone, B alone, union, and the residual with NEITHER -- overall and
split by signed / 4-year vs JUCO, so we see where the gap concentrates.

Read-only.

Run:  python scripts/college_hs_gap.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_hsplace(p):
    p = str(p).strip()
    if ',' in p and p not in ('--,--', 'nan'):
        c, s = p.rsplit(',', 1)
        return (c.strip(), s.strip())
    return None


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    if not os.path.exists(REG):
        print(f"ERROR: {REG} not found.")
        sys.exit(1)
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    FOURYR = ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA']
    JUCO = ['NJCAA', 'CCCAA', 'NWAACC']
    reg['is_college'] = reg['schoolDivision'].astype(str).isin(FOURYR + JUCO)

    # Route A: backlink HS ids (any year)
    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    backlink_ids = set(hs_rows.loc[hs_rows['loc'].notna(), 'PlayerID'].unique())

    # Route B: hsplace ids
    hsplace_ids = set()
    if os.path.exists(SB):
        sb = pd.read_csv(SB, low_memory=False)
        sb_pid = find(sb, 'playerid', 'PlayerID')
        hsplace = find(sb, 'hsplace')
        if sb_pid and hsplace:
            for _, r in sb.iterrows():
                if parse_hsplace(r[hsplace]):
                    hsplace_ids.add(r[sb_pid])

    # college players 1996-2019, deduped
    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    players = col.drop_duplicates('PlayerID').copy()
    n = len(players)

    players['A'] = players['PlayerID'].isin(backlink_ids)
    players['B'] = players['PlayerID'].isin(hsplace_ids)
    players['either'] = players['A'] | players['B']
    signed = find(reg, 'signed')
    players['signed'] = players[signed].astype(str).str.strip().isin(
        ['Y', '1', 'Yes'])
    players['bucket'] = players['schoolDivision'].apply(
        lambda d: '4-year' if d in FOURYR else ('JUCO' if d in JUCO else 'other'))

    a, b = int(players['A'].sum()), int(players['B'].sum())
    either = int(players['either'].sum())
    neither = n - either

    print("="*62)
    print(f"COLLEGE DRAFTEES 1996-2019 (deduped): {n:,}")
    print("="*62)
    print(f"  A. backlink (own HS-draft row): {a:>6,}  ({a/n*100:>4.1f}%)")
    print(f"  B. hsplace (signing file):      {b:>6,}  ({b/n*100:>4.1f}%)")
    print(f"  EITHER route -> HS location:    {either:>6,}  ({either/n*100:>4.1f}%)")
    print(f"  NEITHER (no HS location):       {neither:>6,}  ({neither/n*100:>4.1f}%)")
    print(f"\n  >>> {neither/n*100:.1f}% of college draftees have NO deducible HS location <<<")

    print("\n" + "-"*62)
    print("Where the gap concentrates (share WITH a HS location):")
    print("-"*62)
    for col_name, groups in [('signed', [True, False]),
                             ('bucket', ['4-year', 'JUCO'])]:
        for g in groups:
            sub = players[players[col_name] == g]
            if len(sub) == 0:
                continue
            cov = int(sub['either'].sum())
            lab = f"{col_name}={g}"
            print(f"  {lab:<16}: {cov:>6,}/{len(sub):<6,} = {cov/len(sub)*100:>4.1f}% covered"
                  f"   ({len(sub)-cov:,} missing)")

    print("\nNOTE: birth city (place) is ~100% present; the gap is purely HS side.")


if __name__ == '__main__':
    main()
