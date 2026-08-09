"""
Third route to HS location for college draftees: a player drafted out of HS
FIRST (any year, signed or not) has an earlier register row whose `school` field
holds his HS (parsed to city/state). Link that back to his college row by
PlayerID. Measure the coverage this gives.

For each college draftee (1996-2019), check: does this same PlayerID ALSO appear
anywhere in the register (any year) as a HS-draft row with a parseable HS city?

Read-only.

Run:  python scripts/diag_hs_backlink.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'


def parse_school_city(s):
    """HS rows look like 'Name (City,ST)'. Return (city, state) or None."""
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None


def main():
    if not os.path.exists(REG):
        print(f"ERROR: {REG} not found.")
        sys.exit(1)
    reg = pd.read_csv(REG, low_memory=False)

    sdiv = 'schoolDivision'
    reg['is_hs'] = reg[sdiv].astype(str) == 'HS'
    reg['is_college'] = reg[sdiv].astype(str).isin(
        ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA', 'NJCAA', 'CCCAA', 'NWAACC'])

    # HS location parsed from any HS row (ALL years, not just 1996-2019)
    hs_rows = reg[reg['is_hs']].copy()
    hs_rows['hs_loc'] = hs_rows['school'].apply(parse_school_city)
    hs_rows = hs_rows[hs_rows['hs_loc'].notna()]
    # map PlayerID -> HS (city,state); if multiple, take first
    hs_by_pid = {}
    for _, r in hs_rows.iterrows():
        pid = r['PlayerID']
        if pid not in hs_by_pid:
            hs_by_pid[pid] = r['hs_loc']
    print(f"HS-draft rows with a parseable HS location (all years): "
          f"{len(hs_rows):,}")
    print(f"Distinct PlayerIDs with a HS location: {len(hs_by_pid):,}\n")

    # college draftees 1996-2019
    col = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
              reg['is_college']].copy()
    # a college draftee may have several college rows; dedup to players
    col_players = col.drop_duplicates('PlayerID')
    n_players = len(col_players)

    col_players['has_backlink'] = col_players['PlayerID'].isin(hs_by_pid.keys())
    covered = int(col_players['has_backlink'].sum())

    print("="*64)
    print("HS-LOCATION COVERAGE FOR COLLEGE DRAFTEES via backlink")
    print("="*64)
    print(f"  distinct college draftees (1996-2019): {n_players:,}")
    print(f"  with a HS-draft backlink (own earlier HS row): "
          f"{covered:,} ({covered/n_players*100:.1f}%)")
    print(f"  WITHOUT one (never drafted from HS): "
          f"{n_players-covered:,} ({(n_players-covered)/n_players*100:.1f}%)")

    # birth city is on the college row already (place). how many have BOTH?
    place = 'place'
    col_players['has_birth'] = col_players[place].notna() & \
        (col_players[place].astype(str).str.strip() != '')
    both = int((col_players['has_backlink'] & col_players['has_birth']).sum())
    print(f"\n  with BOTH birth (place) AND backlinked HS: "
          f"{both:,} ({both/n_players*100:.1f}%)")
    print("  ^ these college players could join the SAME birth->HS mover")
    print("    analysis as HS draftees, no signing bias.")

    # combine with hsplace? (union of the two routes) -- rough sizing
    print("\n" + "="*64)
    print("Compare to hsplace route (~37%, signing-biased):")
    print(f"  backlink route: {covered/n_players*100:.1f}% (no signing bias)")
    print("  If backlink >> hsplace, this is the better route. If small, the")
    print("  two could be UNIONED to maximize coverage.")


if __name__ == '__main__':
    main()
