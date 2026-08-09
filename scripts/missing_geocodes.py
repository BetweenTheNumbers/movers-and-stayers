"""
Step 4 (diagnostic) — Find (city, state) pairs not yet successfully geocoded.

Inputs:  v3_analysis.csv, city_coords_v3.csv, geocode_cache_v3.json (optional)
Outputs: cities_to_geocode.csv, geocode_status_report.txt
"""

import pandas as pd
import json
import os
from collections import Counter

ANALYSIS_FILE = 'v3_analysis.csv'
COORDS_FILE   = 'city_coords_v3.csv'
CACHE_FILE    = 'geocode_cache_v3.json'
OUT_CSV  = 'cities_to_geocode.csv'
OUT_TXT  = 'geocode_status_report.txt'

def norm_key(city, state):
    if pd.isna(city) or pd.isna(state):
        return None
    c, s = str(city).strip(), str(state).strip()
    if not c or not s:
        return None
    return f"{c}|{s}"

df = pd.read_csv(ANALYSIS_FILE, low_memory=False)
print(f"Loaded {len(df)} rows from {ANALYSIS_FILE}")

birth_city_col = 'birth_city' if 'birth_city' in df.columns else None
birth_state_col = 'birth_state' if 'birth_state' in df.columns else None
hs_city_col  = 'hs_city'  if 'hs_city'  in df.columns else ('school_city'  if 'school_city'  in df.columns else None)
hs_state_col = 'hs_state' if 'hs_state' in df.columns else ('school_state' if 'school_state' in df.columns else None)

birth_counter, hs_counter = Counter(), Counter()
if birth_city_col and birth_state_col:
    for c, s in zip(df[birth_city_col], df[birth_state_col]):
        k = norm_key(c, s)
        if k: birth_counter[k] += 1
if hs_city_col and hs_state_col:
    for c, s in zip(df[hs_city_col], df[hs_state_col]):
        k = norm_key(c, s)
        if k: hs_counter[k] += 1

all_keys = set(birth_counter) | set(hs_counter)
print(f"  Unique (city,state) pairs: {len(all_keys)}")

if os.path.exists(COORDS_FILE):
    coords = pd.read_csv(COORDS_FILE)
    coords['key'] = coords.apply(lambda r: norm_key(r.get('city'), r.get('state')), axis=1)
    coords['has_coords'] = coords['lat'].notna() & coords['lon'].notna()
    good_keys = set(coords.loc[coords['has_coords'], 'key'].dropna())
    bad_keys = set(coords.loc[~coords['has_coords'], 'key'].dropna())
    all_in_file = set(coords['key'].dropna())
    print(f"\nCoord file: {len(coords)} entries, {len(good_keys)} good, {len(bad_keys)} failed")
else:
    print(f"WARNING: {COORDS_FILE} not found")
    good_keys, bad_keys, all_in_file = set(), set(), set()

missing_entirely = all_keys - all_in_file
failed_in_file = all_keys & bad_keys
to_geocode_keys = missing_entirely | failed_in_file

print(f"\n{'='*70}\nGAP ANALYSIS\n{'='*70}")
print(f"  Total unique pairs:           {len(all_keys):>6}")
print(f"  Already geocoded:             {len(all_keys & good_keys):>6}")
print(f"  STILL NEEDED:                 {len(to_geocode_keys):>6}")
print(f"    - missing entirely:         {len(missing_entirely):>6}")
print(f"    - in file but no lat/lon:   {len(failed_in_file):>6}")

rows = []
for k in to_geocode_keys:
    city, state = k.split('|', 1)
    b, h = birth_counter.get(k, 0), hs_counter.get(k, 0)
    source = 'both' if b and h else ('birth' if b else 'hs')
    rows.append({'city': city, 'state': state, 'used_as_birth_city_count': b,
                 'used_as_hs_city_count': h, 'total_uses': b + h, 'source': source,
                 'reason': 'missing' if k in missing_entirely else 'failed_lookup'})
out_df = pd.DataFrame(rows)
if len(out_df) > 0:
    out_df = out_df.sort_values(['total_uses', 'state', 'city'], ascending=[False, True, True])
    out_df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\nSaved: {OUT_CSV} ({len(out_df)} pairs)")
    print(f"\nMissing by state (top 15):")
    print(out_df.groupby('state').size().sort_values(ascending=False).head(15).to_string())
else:
    pd.DataFrame(columns=['city','state','used_as_birth_city_count','used_as_hs_city_count',
                          'total_uses','source','reason']).to_csv(OUT_CSV, index=False)
    print(f"\nNothing missing — all pairs geocoded.")

with open(OUT_TXT, 'w') as f:
    f.write("Geocoding Gap Report\n" + "="*60 + "\n\n")
    f.write(f"Unique pairs: {len(all_keys)}\n")
    f.write(f"  Geocoded:    {len(all_keys & good_keys)}\n")
    f.write(f"  Still needed:{len(to_geocode_keys)}\n")
    if len(out_df) > 0:
        f.write("\nTop 30 missing by usage:\n")
        f.write(out_df.head(30).to_string(index=False))
print(f"Saved: {OUT_TXT}")
