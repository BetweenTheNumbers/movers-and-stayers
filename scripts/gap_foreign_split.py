"""
How much of the college HS-location gap is FOREIGN-born? The birth->HS mover
variable is cleanest for US players; foreign-born players (came to US college,
never a US HS draftee) are both hard to fill AND a different phenomenon. This
splits the 10,718 gap by US vs non-US birth.

Read-only.

Run:  python scripts/gap_foreign_split.py
"""
import re
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'}


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_place(p):
    p = str(p).strip()
    return ',' in p and p not in ('--,--', 'nan')


def state_of(place):
    p = str(place).strip()
    if ',' in p:
        return p.rsplit(',', 1)[1].strip().upper()
    return ''


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
        ['NCAA 1','NCAA 2','NCAA 3','NAIA','NJCAA','CCCAA','NWAACC'])

    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    backlink_ids = set(hs_rows.loc[hs_rows['loc'].notna(), 'PlayerID'].unique())

    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsp = find(sb, 'hsplace')
    hsplace_ids = set(sb.loc[sb[hsp].apply(parse_place), sb_pid])
    covered = backlink_ids | hsplace_ids

    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    players = col.drop_duplicates('PlayerID').copy()
    gap = players[~players['PlayerID'].isin(covered)].copy()

    gap['bstate'] = gap['place'].apply(state_of)
    gap['foreign'] = ~gap['bstate'].isin(US_STATES)

    n = len(gap)
    f = int(gap['foreign'].sum())
    print("="*56)
    print(f"COLLEGE HS-LOCATION GAP: {n:,} players")
    print("="*56)
    print(f"  US-born:      {n-f:>6,}  ({(n-f)/n*100:.1f}%)")
    print(f"  foreign-born: {f:>6,}  ({f/n*100:.1f}%)")
    print("\n  Top non-US birth 'states'/codes in the gap:")
    for code, cnt in gap.loc[gap['foreign'], 'bstate'].value_counts().head(12).items():
        print(f"    {code:<6} {cnt:,}")

    # of the whole college pop, how many are foreign (context)
    players['bstate'] = players['place'].apply(state_of)
    players['foreign'] = ~players['bstate'].isin(US_STATES)
    pf = int(players['foreign'].sum())
    print(f"\n  For context: {pf:,}/{len(players):,} "
          f"({pf/len(players)*100:.1f}%) of ALL college draftees are foreign-born.")
    # what share of foreign are in the gap vs US share in gap
    us = players[~players['foreign']]
    fo = players[players['foreign']]
    us_gap = (~us['PlayerID'].isin(covered)).mean()*100
    fo_gap = (~fo['PlayerID'].isin(covered)).mean()*100
    print(f"  US-born college players missing HS loc:      {us_gap:.1f}%")
    print(f"  foreign-born college players missing HS loc: {fo_gap:.1f}%")
    print("\n  READ: if foreign-born are mostly in the gap, much of what's")
    print("  'missing' is players the US birth->HS mover variable doesn't")
    print("  cleanly describe anyway.")


if __name__ == '__main__':
    main()
