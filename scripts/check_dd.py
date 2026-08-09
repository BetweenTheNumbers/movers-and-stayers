"""
Do we have HS location for college draftees?

General answer: no -- a college draftee's 'school' field is his college, and
there's no separate HS field. BUT a player drafted out of HS first (didn't sign)
and later out of college has TWO register rows, and his HS row carries his HS
location. This checks how many such players exist and whether we can assemble
all three anchors (birth, HS, college) for them by joining their two rows.

Run:  python scripts/check_dd.py
"""
import pandas as pd

df = pd.read_csv('v3_analysis_with_war.csv', low_memory=False)

g = df[df['mlbid'].notna()].groupby('mlbid')['is_hs_draftee'].nunique()
both_ids = g[g > 1].index
dd = df[df['mlbid'].isin(both_ids)]
print(f"Players drafted as BOTH HS and college: {dd['mlbid'].nunique():,}")

hs_rows = dd[dd['is_hs_draftee'] == 1]
col_rows = dd[dd['is_hs_draftee'] == 0]

hs_loc = (hs_rows['birth_city'].notna() & hs_rows['hs_city'].notna())
print(f"  their HS rows with birth+HS location:     "
      f"{int(hs_loc.sum()):,} / {len(hs_rows):,}")
print(f"  their college rows (any):                 {len(col_rows):,}")

# college row: does school_city hold the college location?
col_loc = (col_rows['birth_city'].notna() & col_rows['school_city'].notna())
print(f"  their college rows with birth+college loc: "
      f"{int(col_loc.sum()):,} / {len(col_rows):,}")

# players where the HS row has HS loc AND a college row exists with college loc
hs_ok_ids = set(hs_rows[hs_loc]['mlbid'])
col_ok_ids = set(col_rows[col_loc]['mlbid'])
all3 = hs_ok_ids & col_ok_ids
print(f"\n  Players with ALL THREE anchors available "
      f"(birth + HS + college): {len(all3):,}")
print("  For THESE players we could compute birth->HS, HS->college, AND")
print("  birth->college. For college-only draftees we cannot (no HS row).")

# how many college draftees are college-ONLY (never drafted from HS)?
col_only = df[(df['is_hs_draftee'] == 0) &
              (~df['mlbid'].isin(both_ids))]
print(f"\n  College-only draftees (no HS row, HS location unavailable): "
      f"{col_only['mlbid'].nunique():,}")
