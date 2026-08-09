"""
Measure the UNION of the two no-new-data routes to HS location for college
draftees (1996-2019):
  A. backlink: player's own earlier HS-draft row (register, any year) -> HS city
  B. hsplace:  signing-bonus file hsplace field
Reports each alone and combined, plus the signing-bias check on the union.

Read-only. (Clean .loc usage, no SettingWithCopy warnings.)

Run:  python scripts/diag_hs_union.py
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


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    reg['is_college'] = reg['schoolDivision'].astype(str).isin(
        ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA', 'NJCAA', 'CCCAA', 'NWAACC'])

    # route A: backlink HS city by PlayerID (all years)
    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['hs_loc'] = hs_rows['school'].apply(parse_school_city)
    hs_rows = hs_rows.loc[hs_rows['hs_loc'].notna()]
    backlink_ids = set(hs_rows['PlayerID'].unique())

    # route B: hsplace ids from signing-bonus
    hsplace_ids = set()
    if os.path.exists(SB):
        sb = pd.read_csv(SB, low_memory=False)
        sb_pid = find(sb, 'playerid', 'PlayerID')
        hsplace = find(sb, 'hsplace')
        if sb_pid and hsplace:
            good = sb.loc[sb[hsplace].notna()].copy()
            good['ok'] = (good[hsplace].astype(str).str.strip()
                          .replace('nan', '').replace('--,--', '') != '')
            hsplace_ids = set(good.loc[good['ok'], sb_pid].unique())

    # college players 1996-2019, deduped
    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    players = col.drop_duplicates('PlayerID').copy()
    n = len(players)

    players['A_backlink'] = players['PlayerID'].isin(backlink_ids)
    players['B_hsplace'] = players['PlayerID'].isin(hsplace_ids)
    players['either'] = players['A_backlink'] | players['B_hsplace']
    players['both'] = players['A_backlink'] & players['B_hsplace']

    a = int(players['A_backlink'].sum())
    b = int(players['B_hsplace'].sum())
    either = int(players['either'].sum())
    both = int(players['both'].sum())

    print("="*64)
    print("HS-LOCATION COVERAGE FOR COLLEGE DRAFTEES (1996-2019)")
    print("="*64)
    print(f"  distinct college players: {n:,}\n")
    print(f"  A. backlink (own HS row):   {a:>6,}  ({a/n*100:>4.1f}%)")
    print(f"  B. hsplace (signing file):  {b:>6,}  ({b/n*100:>4.1f}%)")
    print(f"  both A and B:               {both:>6,}  ({both/n*100:>4.1f}%)")
    print(f"  EITHER (union):             {either:>6,}  ({either/n*100:>4.1f}%)")
    print(f"  neither (no HS loc):        {n-either:>6,}  "
          f"({(n-either)/n*100:>4.1f}%)")

    # signing-bias check on the union
    signed = find(reg, 'signed')
    players['is_signed'] = players[signed].astype(str).str.strip().isin(
        ['Y', '1', 'Yes'])
    print("\n" + "="*64)
    print("Signing-bias check on the UNION")
    print("="*64)
    for lab, m in [('signed', players['is_signed']),
                   ('unsigned', ~players['is_signed'])]:
        sub = players.loc[m]
        cov = int(sub['either'].sum())
        print(f"  {lab:<10}: {cov:,}/{len(sub):,} = {cov/max(len(sub),1)*100:.1f}% "
              f"have HS loc (union)")
    print("\n  If signed vs unsigned union-coverage is CLOSE, the selection")
    print("  problem is largely solved. If still far apart, bias remains.")


if __name__ == '__main__':
    main()
