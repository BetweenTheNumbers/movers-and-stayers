"""
How much geocoding to add 1996-2019 COLLEGE players?

The work is driven by DISTINCT locations, not player count -- a name->city
crosswalk built once covers every player at that school. This counts:
  - college players 1996-2019 (by schoolDivision bucket)
  - DISTINCT college names to resolve to a city
  - DISTINCT birth places, and how many are already in the geocode cache
  - how many distinct colleges cover 90% / 95% of players (the crosswalk can be
    built high-value-first)

Run:  python scripts/count_college_geocode.py
"""
import os
import pandas as pd

REG = 'data/tbc_draft_register.csv'
CACHE = 'geocode_cache_v3.json'

df = pd.read_csv(REG, low_memory=False)
df = df[(df['year'] >= 1996) & (df['year'] <= 2019)].copy()
print(f"1996-2019 draft records: {len(df):,}")

div = df['schoolDivision'].astype(str)
is_hs = div.eq('HS')
is_juco = div.isin(['NJCAA', 'CCCAA', 'NWAACC'])
is_4yr = div.isin(['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA'])
is_college = is_juco | is_4yr

print(f"  HS:        {int(is_hs.sum()):,}")
print(f"  4-year:    {int(is_4yr.sum()):,}")
print(f"  JUCO:      {int(is_juco.sum()):,}")
print(f"  college total: {int(is_college.sum()):,}")

col = df[is_college].copy()

# ---- distinct colleges to resolve ----
# college 'school' is a bare name (no city); normalize for counting
col['school_norm'] = col['school'].astype(str).str.strip().str.lower()
distinct_schools = col['school_norm'].nunique()
print("\n" + "="*60)
print(f"DISTINCT college names to map -> city: {distinct_schools:,}")
print("="*60)
vc = col['school_norm'].value_counts()
# how many distinct schools cover 90 / 95% of college players?
cum = vc.cumsum() / vc.sum()
for pct in [0.90, 0.95, 0.99]:
    n = int((cum <= pct).sum()) + 1
    print(f"  top {n:,} schools cover {pct*100:.0f}% of college players")
print(f"\n  Top 15 colleges by draftee count:")
for name, n in vc.head(15).items():
    print(f"    {n:>4}  {name}")

# ---- distinct birth places + cache coverage ----
print("\n" + "="*60)
print("BIRTH places for college players (also need geocoding)")
print("="*60)
# cache keys are "city|state" (see v3_geocode.py). Build the same from 'place'
# which is formatted "City,ST".
def place_to_key(p):
    p = str(p).strip()
    if ',' in p:
        city, state = p.rsplit(',', 1)
        return f"{city.strip()}|{state.strip()}"
    return p

col['place_key'] = col['place'].apply(place_to_key)
distinct_births = col['place_key'].nunique()
print(f"  distinct birth places (college players): {distinct_births:,}")
if os.path.exists(CACHE):
    import json
    cache = json.load(open(CACHE))
    cache_keys = set(cache.keys())
    distinct_cached = sum(1 for p in col['place_key'].unique() if p in cache_keys)
    print(f"  geocode cache entries: {len(cache):,}")
    print(f"  college birth places already in cache (distinct): "
          f"{distinct_cached:,} / {distinct_births:,}")
    print(f"  -> NEW distinct birth places to geocode: "
          f"{distinct_births - distinct_cached:,}")
    # sanity: show a couple of matched + unmatched keys
    matched = [p for p in col['place_key'].unique() if p in cache_keys][:3]
    unmatched = [p for p in col['place_key'].unique() if p not in cache_keys][:3]
    print(f"  sample matched keys:   {matched}")
    print(f"  sample unmatched keys: {unmatched}")
else:
    print(f"  (cache {CACHE} not found -- all {distinct_births:,} would be new)")

print("\n" + "="*60)
print("BOTTOM LINE")
print("="*60)
print(f"  College players to add (1996-2019): {int(is_college.sum()):,}")
print(f"  Distinct colleges to resolve to a city: {distinct_schools:,}")
print(f"  New birth places to geocode: see above")
print("  Colleges resolve via a NAME->CITY crosswalk (once per school), then")
print("  the city geocodes like any other. Work scales with DISTINCT schools,")
print("  not player count.")
