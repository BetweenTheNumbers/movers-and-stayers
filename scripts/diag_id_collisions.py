"""
Detect PlayerIDs that likely conflate MULTIPLE real players (identity collision),
which corrupts birth/HS/draft linkage. Signals per PlayerID:
  - more than one distinct BIRTH place (`place`)
  - birth dates (borndate) that differ by more than a few years
  - more than one distinct HS location across its HS rows
  - draft years spanning implausibly long (> ~10 yrs HS->college is suspicious)

Prints the worst offenders with their conflicting rows so they can be verified /
split manually. Writes college_id_collisions.csv.

Read-only.

Run:  python scripts/diag_id_collisions.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    if m:
        return f"{m.group(1).strip()},{m.group(2).strip()}"
    return None


def yr_of(bd):
    """extract a 4-digit year from a borndate string."""
    m = re.search(r'(19|20)\d\d', str(bd))
    return int(m.group(0)) if m else None


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    reg['is_college'] = reg['schoolDivision'].astype(str).isin(
        ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA', 'NJCAA', 'CCCAA', 'NWAACC'])

    # PlayerIDs that are actually IN scope: appear as a college draftee 1996-2019
    in_scope = set(reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                           reg['is_college'], 'PlayerID'].unique())

    has_born = 'borndate' in reg.columns
    collisions = []
    for pid, g in reg.groupby('PlayerID'):
        if pid not in in_scope:
            continue  # only care about players in the college analysis set
        places = set(str(p).strip() for p in g['place'].dropna()
                     if str(p).strip() not in ('', 'nan', '--,--'))
        years = set()
        if has_born:
            years = set(y for y in (yr_of(b) for b in g['borndate']) if y)
        hslocs = set(filter(None, (parse_school_city(s)
                                   for s in g.loc[g['is_hs'], 'school'])))
        draft_years = sorted(int(y) for y in g['year'].dropna().unique())

        # classify
        multi_birth = len(places) > 1
        birthyr_span = (len(years) > 1 and (max(years) - min(years)) > 3)
        draftyr_span = (draft_years and (max(draft_years) - min(draft_years)) > 12)
        multi_hs = len(hslocs) > 1

        # GENUINE collision = multiple people signals
        genuine = multi_birth or birthyr_span or draftyr_span
        # BENIGN = single person, just HS labeled 2 ways (often one matches birth)
        benign = multi_hs and not genuine

        if genuine or benign:
            reasons = []
            if multi_birth:
                reasons.append(f"{len(places)} birthplaces")
            if birthyr_span:
                reasons.append(f"birthyr span {max(years)-min(years)}y")
            if draftyr_span:
                reasons.append(f"draftyr span {max(draft_years)-min(draft_years)}y")
            if multi_hs:
                reasons.append(f"{len(hslocs)} HS locs")
            collisions.append({
                'PlayerID': pid,
                'kind': 'GENUINE' if genuine else 'benign',
                'names': ' / '.join(sorted(set(
                    f"{a} {b}" for a, b in
                    zip(g['firstName'].astype(str), g['lastName'].astype(str))))),
                'n_rows': len(g),
                'birth_places': ' | '.join(sorted(places)),
                'birth_years': ','.join(str(y) for y in sorted(years)),
                'hs_locations': ' | '.join(sorted(hslocs)),
                'draft_years': ','.join(str(y) for y in draft_years),
                'why': '; '.join(reasons),
            })

    cdf = pd.DataFrame(collisions)
    if len(cdf):
        cdf = cdf.sort_values(['kind', 'n_rows'], ascending=[True, False])
        cdf.to_csv('college_id_collisions.csv', index=False)

    n_gen = int((cdf['kind'] == 'GENUINE').sum()) if len(cdf) else 0
    n_ben = int((cdf['kind'] == 'benign').sum()) if len(cdf) else 0
    print("="*74)
    print(f"IN-SCOPE (1996-2019 college) id issues: "
          f"{n_gen} GENUINE collisions, {n_ben} benign HS-variants")
    print("="*74)
    print("GENUINE = 2+ birthplaces / far-apart birth or draft years = MULTIPLE")
    print("people under one id. These corrupt linkage; exclude or split.\n")
    for _, r in cdf.loc[cdf['kind'] == 'GENUINE'].head(40).iterrows():
        print(f"PID {r['PlayerID']}  [{r['why']}]  drafts:{r['draft_years']}")
        print(f"    {r['names']}")
        print(f"    births: {r['birth_places']}  HS: {r['hs_locations']}")
    print(f"\nbenign HS-variants (single person, 2 HS labels): {n_ben} "
          f"-- pick the birth-matching or later one; low risk")
    print(f"\nFull list -> college_id_collisions.csv")
    print("GENUINE ones are what need manual verify/exclude before building.")


if __name__ == '__main__':
    main()
