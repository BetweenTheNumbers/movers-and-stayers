"""
The 25 HIGHEST-drafted college players (1996-2019) who have NO deducible HS
location (no HS-draft backlink, no hsplace). Sorted by overall pick so the most
recognizable names are on top -- for manual eyeballing.

Read-only.

Run:  python scripts/gap_top_picks.py
"""
import re
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_place(p):
    p = str(p).strip()
    return ',' in p and p not in ('--,--', 'nan')


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

    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    backlink_ids = set(hs_rows.loc[hs_rows['loc'].notna(), 'PlayerID'].unique())

    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsp = find(sb, 'hsplace')
    hsname = find(sb, 'hsName', 'hsname')
    hsplace_ids = set()
    hsname_by_pid = {}
    for _, r in sb.iterrows():
        if parse_place(r[hsp]):
            hsplace_ids.add(r[sb_pid])
        if hsname and str(r[hsname]).strip() not in ('', 'nan', '-'):
            hsname_by_pid.setdefault(r[sb_pid], str(r[hsname]).strip())
    covered = backlink_ids | hsplace_ids

    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    # for a player with multiple college rows, keep his BEST (lowest overall) pick
    col = col.sort_values('overall')
    col_players = col.drop_duplicates('PlayerID', keep='first')
    gap = col_players[~col_players['PlayerID'].isin(covered)].copy()

    gap = gap.sort_values('overall').head(25)

    print(f"25 highest-drafted college players with NO HS location "
          f"(of {(~col_players['PlayerID'].isin(covered)).sum():,} total gap):\n")
    print(f"{'overall':>7} {'yr':>5} {'rd':>3}  {'name':<24}{'college':<26}"
          f"{'born':<20} hsName?")
    print("-"*104)
    for _, r in gap.iterrows():
        nm = f"{r['firstName']} {r['lastName']}"
        hn = hsname_by_pid.get(r['PlayerID'], '')
        ov = int(r['overall']) if pd.notna(r['overall']) else 0
        rd = int(r['draftRound']) if pd.notna(r['draftRound']) else 0
        print(f"{ov:>7} {int(r['year']):>5} {rd:>3}  {nm:<24}"
              f"{str(r['school'])[:25]:<26}{str(r['place']):<20}{hn}")

    print("\n(hsName shown when the signing file has a HS NAME but no city -- "
          "those are resolvable.)")


if __name__ == '__main__':
    main()
