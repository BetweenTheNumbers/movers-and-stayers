"""
Produce the list of backlink-vs-hsplace HS-location conflicts that are NOT
confidently explainable, for manual lookup. Auto-classifies and sets aside:
  - SAME-CITY, different state (typo pairs like Mexico MO/MS, Solon IA/OH)
  - one side CANADA / non-US (prep-school-vs-hometown, keep US side later)
Everything else = genuine conflict -> printed + written to
college_hs_conflicts_to_check.csv with player name, year, round, birth place,
division, and BOTH candidate HS locations, so they can be looked up.

Read-only.

Run:  python scripts/list_hs_conflicts.py
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
    c = re.sub(r'\s+', ' ', c)
    c = c.replace('saint ', 'st ').replace('ft ', 'fort ')
    return c


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'

    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    hs_rows = hs_rows.loc[hs_rows['loc'].notna()]
    backlink = {}
    for _, r in hs_rows.iterrows():
        if r['PlayerID'] not in backlink:
            backlink[r['PlayerID']] = r['loc']

    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsplace = find(sb, 'hsplace')
    hsp = {}
    for _, r in sb.iterrows():
        loc = parse_hsplace(r[hsplace])
        if loc and r[sb_pid] not in hsp:
            hsp[r[sb_pid]] = loc

    # reg lookup for player details
    reg_by_pid = reg.drop_duplicates('PlayerID').set_index('PlayerID')

    both = set(backlink) & set(hsp)
    genuine = []
    same_city_typo = 0
    canada_case = 0
    for pid in both:
        bc, bs = backlink[pid]
        hc, hs = hsp[pid]
        if bs.upper() == hs.upper():
            continue  # states agree, not a conflict
        # auto-explain: same city, different state
        if norm_city(bc) == norm_city(hc):
            same_city_typo += 1
            continue
        # auto-explain: one side is Canada / non-US
        if bs.upper() in CA_PROVS or hs.upper() in CA_PROVS or \
           bs.upper() not in US_STATES or hs.upper() not in US_STATES:
            canada_case += 1
            continue
        genuine.append((pid, bc, bs, hc, hs))

    print(f"Total state conflicts: {len(genuine)+same_city_typo+canada_case}")
    print(f"  auto-explained same-city/diff-state typos: {same_city_typo}")
    print(f"  auto-explained Canada/non-US cases:        {canada_case}")
    print(f"  GENUINE conflicts needing lookup:          {len(genuine)}\n")

    rows = []
    print("="*76)
    print("LOOK THESE UP -- which HS location is correct? (backlink vs hsplace)")
    print("="*76)
    for pid, bc, bs, hc, hs in genuine:
        r = reg_by_pid.loc[pid] if pid in reg_by_pid.index else None
        name = f"{r['firstName']} {r['lastName']}" if r is not None else '?'
        yr = int(r['year']) if r is not None and pd.notna(r['year']) else 0
        rnd = int(r['draftRound']) if r is not None and pd.notna(r['draftRound']) else 0
        born = str(r['place']) if r is not None else '?'
        div = r['schoolDivision'] if r is not None else '?'
        print(f"  PID {pid}  {name:<22} {yr} rd{rnd:<3} born:{born:<18}")
        print(f"       backlink HS: {bc},{bs:<4}   hsplace HS: {hc},{hs}")
        rows.append({'PlayerID': pid, 'player': name, 'year': yr, 'round': rnd,
                     'birth_place': born, 'division': div,
                     'backlink_hs': f"{bc},{bs}", 'hsplace_hs': f"{hc},{hs}",
                     'correct_which': ''})

    pd.DataFrame(rows).to_csv('college_hs_conflicts_to_check.csv', index=False)
    print(f"\nSaved {len(rows)} rows to college_hs_conflicts_to_check.csv")
    print("Fill 'correct_which' with backlink / hsplace / other and send back.")


if __name__ == '__main__':
    main()
