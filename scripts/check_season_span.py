"""
Quick span check before building the 12-year-window rate analysis.

Tells us: how recent the FanGraphs season data goes, and how many MLB-reaching
HS draftees fall in draft years old enough to have a COMPLETE 12-year window.

Run:  python scripts/check_season_span.py
"""

import pandas as pd

max_season = 0
for f in ['fg-hit-seasons.csv', 'fg-pit-seasons.csv']:
    d = pd.read_csv(f, low_memory=False)
    lo, hi = int(d['Season'].min()), int(d['Season'].max())
    print(f"{f}: Season range {lo}-{hi}")
    max_season = max(max_season, hi)

print(f"\nLatest season in data: {max_season}")

a = pd.read_csv('v3_analysis_with_war.csv', low_memory=False)
a = a[(a['is_hs_draftee'] == 1) & (a['reached_mlb'] == 1)]
print(f"MLB-reaching HS draftees: {len(a):,}")
print(f"Draft year range: {int(a['year'].min())}-{int(a['year'].max())}")

cutoff = max_season - 12
print(f"\nA full 12-year window (draft+1 .. draft+12) needs draft year <= {cutoff}")
unc = int((a['year'] <= cutoff).sum())
cen = int((a['year'] > cutoff).sum())
print(f"  Uncensored (full window available): {unc:,}")
print(f"  Censored (window still open):       {cen:,}")

print("\nMLB reachers by draft year:")
print(a['year'].value_counts().sort_index().to_string())
