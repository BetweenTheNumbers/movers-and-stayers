"""
Does the 1996 cutoff on the BACKLINK matter? A college player (drafted 1996-2019)
might get his HS location from an HS-draft row that itself was pre-1996. Measure:
  1. Of college players with a backlink, how many have their EARLIEST HS row
     pre-1996 vs 1996+?
  2. Of the genuine state-conflicts, how many involve a pre-1996 HS row?
This tells us whether restricting backlink to 1996+ changes coverage/conflicts
meaningfully.

Read-only.

Run:  python scripts/diag_backlink_year.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'

CA_PROVS = {'ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE',
            'YT', 'NT', 'NU'}
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'DC',
}


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
    c = re.sub(r"[.'`]", '', c)
    return re.sub(r'\s+', ' ', c)


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    reg['is_college'] = reg['schoolDivision'].astype(str).isin(
        ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA', 'NJCAA', 'CCCAA', 'NWAACC'])

    # HS rows with location + their draft YEAR, keep EARLIEST per player
    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    hs_rows = hs_rows.loc[hs_rows['loc'].notna()]
    hs_rows = hs_rows.sort_values('year')
    backlink = {}
    backlink_year = {}
    for _, r in hs_rows.iterrows():
        pid = r['PlayerID']
        if pid not in backlink:
            backlink[pid] = r['loc']
            backlink_year[pid] = int(r['year'])

    # college players 1996-2019
    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    players = col.drop_duplicates('PlayerID').copy()
    players['bl_year'] = players['PlayerID'].map(backlink_year)
    have_bl = players.loc[players['bl_year'].notna()]

    print("="*60)
    print("BACKLINK YEAR distribution (college players 1996-2019)")
    print("="*60)
    n = len(have_bl)
    pre96 = int((have_bl['bl_year'] <= 1995).sum())
    post = int((have_bl['bl_year'] >= 1996).sum())
    print(f"  college players with a backlink: {n:,}")
    print(f"    HS-draft row 1996+:      {post:,} ({post/n*100:.1f}%)")
    print(f"    HS-draft row <=1995:     {pre96:,} ({pre96/n*100:.1f}%)")
    print(f"  -> restricting backlink to 1996+ would DROP {pre96:,} players'")
    print(f"     backlink (they may still have hsplace).")

    # how many of those pre-96-only would lose HS loc entirely (no hsplace)?
    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsplace = find(sb, 'hsplace')
    hsp_ids = set()
    for _, r in sb.iterrows():
        if parse_hsplace(r[hsplace]):
            hsp_ids.add(r[sb_pid])
    pre96_ids = set(have_bl.loc[have_bl['bl_year'] <= 1995, 'PlayerID'])
    lose = sum(1 for pid in pre96_ids if pid not in hsp_ids)
    print(f"  of those {pre96:,} pre-1996 backlinks, {lose:,} have NO hsplace")
    print(f"     -> would lose HS location entirely if we cut pre-96 backlinks.")

    # conflicts: how many genuine state-conflicts involve a pre-96 HS row?
    hsp = {}
    for _, r in sb.iterrows():
        loc = parse_hsplace(r[hsplace])
        if loc and r[sb_pid] not in hsp:
            hsp[r[sb_pid]] = loc
    both = set(backlink) & set(hsp)
    genuine = 0
    genuine_pre96 = 0
    for pid in both:
        bc, bs = backlink[pid]
        hc, hs = hsp[pid]
        if bs.upper() == hs.upper():
            continue
        if norm_city(bc) == norm_city(hc):
            continue
        if bs.upper() in CA_PROVS or hs.upper() in CA_PROVS or \
           bs.upper() not in US_STATES or hs.upper() not in US_STATES:
            continue
        genuine += 1
        if backlink_year.get(pid, 9999) <= 1995:
            genuine_pre96 += 1
    print("\n" + "="*60)
    print("GENUINE conflicts vs the 1996 cutoff")
    print("="*60)
    print(f"  genuine state-conflicts (all): {genuine}")
    print(f"  ...of which backlink row is pre-1996: {genuine_pre96}")
    print(f"  ...remaining if we ignore pre-96 backlinks: {genuine-genuine_pre96}")


if __name__ == '__main__':
    main()
