"""Dump identifying ids for the 'concordia university' players so we can build a
per-player location override keyed on a stable id.

Run:  python scripts/concordia_ids.py
"""
import pandas as pd
from college_crosswalk import normalize

reg = pd.read_csv('data/tbc_draft_register.csv',
                  low_memory=False)
reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
reg['k'] = reg['school'].apply(normalize)

sub = reg[reg['k'] == 'concordia university'].copy()
print("concordia university players with ids:\n")
print(f"{'mlbid':>8}  {'PlayerID':>9}  {'name':<22}{'yr':>5} {'born':<20} div")
for _, r in sub.sort_values('year').iterrows():
    mlbid = r['mlbid'] if pd.notna(r['mlbid']) else ''
    pid = r['PlayerID'] if 'PlayerID' in r and pd.notna(r['PlayerID']) else ''
    name = f"{r['firstName']} {r['lastName']}"
    print(f"{str(mlbid):>8}  {str(pid):>9}  {name:<22}{int(r['year']):>5} "
          f"{str(r['place']):<20} {r['schoolDivision']}")
