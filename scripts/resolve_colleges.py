"""
Apply the college crosswalk to 1996-2019 college draftees, report coverage, and
write a punch-list of unmatched schools ranked by draftee count.

The built-in crosswalk (college_crosswalk.py) covers the high-value head. This
tells you what % of college PLAYERS are covered (the number that matters, since
a school with 150 draftees matters 150x more than one with 1), and writes the
unmatched names to a CSV template you can fill (city,state) or feed IPEDS into.

Run:  python scripts/resolve_colleges.py
Outputs:
  college_locations_resolved.csv   (school -> city,state for matched)
  college_unmatched.csv            (punch-list: unmatched school, n_draftees)
"""
import os
import sys
import pandas as pd

try:
    from college_crosswalk import CROSSWALK, normalize, lookup, lookup_player
except ImportError:
    print("ERROR: college_crosswalk.py must be in the same scripts dir.")
    sys.exit(1)

REG = 'data/tbc_draft_register.csv'


def main():
    if not os.path.exists(REG):
        print(f"ERROR: {REG} not found.")
        sys.exit(1)
    df = pd.read_csv(REG, low_memory=False)
    df = df[(df['year'] >= 1996) & (df['year'] <= 2019)].copy()

    div = df['schoolDivision'].astype(str)
    is_college = div.isin(['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA',
                           'NJCAA', 'CCCAA', 'NWAACC'])
    col = df[is_college].copy()
    col['key'] = col['school'].apply(normalize)

    total_players = len(col)
    distinct = col['key'].nunique()

    # match: per-player override first (for one-name-many-campus cases), then
    # the school-level crosswalk (exact + alias)
    pidcol = next((c for c in ['PlayerID', 'playerid', 'PlayerId']
                   if c in col.columns), None)

    def resolve_row(r):
        if pidcol is not None:
            ov = lookup_player(r[pidcol])
            if ov is not None:
                return ov
        return lookup(r['school'])

    col['_loc'] = col.apply(resolve_row, axis=1)
    col['matched'] = col['_loc'].notna()
    matched_players = int(col['matched'].sum())
    matched_schools = col[col['matched']]['key'].nunique()

    print("="*64)
    print("COLLEGE CROSSWALK COVERAGE (1996-2019)")
    print("="*64)
    print(f"  college draft records:      {total_players:,}")
    print(f"  distinct college names:     {distinct:,}")
    print(f"  crosswalk entries built:    {len(CROSSWALK):,}")
    print(f"\n  PLAYERS covered:  {matched_players:,} / {total_players:,} "
          f"({matched_players/total_players*100:.1f}%)")
    print(f"  SCHOOLS covered:  {matched_schools:,} / {distinct:,} "
          f"({matched_schools/distinct*100:.1f}%)")
    print("  (player coverage is the number that matters -- big schools first)")

    # resolved output
    matched = col[col['matched']].copy()
    matched['_city'] = matched['_loc'].apply(lambda t: t[0])
    matched['_state'] = matched['_loc'].apply(lambda t: t[1])
    out = (matched[['school', 'key', '_city', '_state']]
           .drop_duplicates('key')
           .rename(columns={'_city': 'city', '_state': 'state'}))
    out.to_csv('college_locations_resolved.csv', index=False)

    # unmatched punch-list, ranked by draftee count
    un = col[~col['matched']]
    punch = (un.groupby('key').size().sort_values(ascending=False)
             .reset_index(name='n_draftees'))
    punch['city'] = ''
    punch['state'] = ''
    punch.to_csv('college_unmatched.csv', index=False)

    print("\n" + "="*64)
    print(f"UNMATCHED: {len(punch):,} schools, {int(punch['n_draftees'].sum()):,} "
          f"draftee records")
    print("="*64)
    print("  Top 25 unmatched by draftee count (fill these first):")
    for _, r in punch.head(25).iterrows():
        print(f"    {int(r['n_draftees']):>4}  {r['key']}")

    # how many MORE schools to hit 95% / 99% of players
    cum_unmatched = punch['n_draftees'].cumsum()
    need_95 = total_players * 0.95 - matched_players
    need_99 = total_players * 0.99 - matched_players
    if need_95 > 0:
        n95 = int((cum_unmatched < need_95).sum()) + 1
        print(f"\n  Fill {n95} more schools -> ~95% player coverage")
    if need_99 > 0:
        n99 = int((cum_unmatched < need_99).sum()) + 1
        print(f"  Fill {n99} more schools -> ~99% player coverage")

    print("\nSaved: college_locations_resolved.csv, college_unmatched.csv")
    print("Fill city,state in college_unmatched.csv for the schools you want,")
    print("then we merge it into the crosswalk and geocode the distinct cities.")


if __name__ == '__main__':
    main()
