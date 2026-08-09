"""How much of the 0-1 mile bin is exact same-city? Clarifies the histogram label."""
import pandas as pd

df = pd.read_csv('v3_analysis_with_war.csv', low_memory=False)
hs = df[df['is_hs_draftee'] == 1].copy()
d = hs['distance_miles']

le1 = hs[d <= 1]
print(f"players with distance <= 1 mile: {len(le1):,}")
if 'same_city' in hs.columns:
    sc = le1['same_city']
    print(f"  of those, same_city flag == 1: {int((sc == 1).sum()):,}")
    print(f"  same_city == 0 (diff name, <1mi apart): {int((sc == 0).sum()):,}")
    print(f"  same_city missing: {int(sc.isna().sum()):,}")
# and exact zeros
print(f"  exactly 0.0 miles: {int((le1['distance_miles'] == 0).sum()):,}")
print(f"  between 0 and 1 (exclusive of 0): {int(((le1['distance_miles']>0)&(le1['distance_miles']<=1)).sum()):,}")
