"""
Step 3 — Fix-and-refill helper for the geocoder gap.

  1. Drops placeholders ('--') and uppercases lowercase state typos.
  2. Applies known typo corrections.
  3. Evicts failed-lookup cache entries so they get re-attempted.
  4. Expands COUNTRY_MAP so foreign state codes (VZ, JM, etc.) resolve.
  5. Re-geocodes anything still missing; updates cache + coords in place.

Inputs:  v3_analysis.csv, city_coords_v3.csv, geocode_cache_v3.json
Outputs: geocode_cache_v3.json, city_coords_v3.csv, refill_report.txt
"""

import pandas as pd

# Shared reference tables — single source of truth (see geo_tables.py)
from geo_tables import (US_STATES, CA_PROVS, COUNTRY_MAP,
                        TYPO_FIXES, MANUAL_COORDS)
import json
import os
import time
import requests

ANALYSIS_FILE = 'v3_analysis.csv'
COORDS_FILE   = 'city_coords_v3.csv'
CACHE_FILE    = 'geocode_cache_v3.json'
REPORT_FILE   = 'refill_report.txt'

NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
HEADERS = {'User-Agent': 'DraftMobilityResearch/1.0 (academic research)'}
TIMEOUT = 20


# NOTE: 'NT' deliberately NOT treated as Northwest Territories. In this dataset
# NT means the Netherlands (Amsterdam, Apeldoorn, Rosmalen all appear with NT).


# Places that are real but absent from (or unfindable in) Nominatim.
# Seeded directly into the cache so they never hit the network.
# Provenance is recorded so these can be audited later.


def normalize_state(s):
    if pd.isna(s):
        return s
    s = str(s).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return s

def norm_key(city, state):
    if pd.isna(city) or pd.isna(state):
        return None
    c, s = str(city).strip(), str(state).strip()
    if not c or not s:
        return None
    return f"{c}|{s}"

def apply_fixes(city, state):
    if pd.isna(city) or pd.isna(state):
        return city, state
    c = str(city).strip()
    s = normalize_state(state)
    if c == '--' or s == '--':
        return None, None
    if (c, s) in TYPO_FIXES:
        return TYPO_FIXES[(c, s)]
    return c, s

def build_query(city, state):
    if not city or not state:
        return None
    if state in US_STATES:
        return f"{city}, {state}, USA"
    if state in CA_PROVS:
        return f"{city}, {state}, Canada"
    if state in COUNTRY_MAP:
        country = COUNTRY_MAP[state]
        if not country:
            return None
        return f"{city}, {country}"
    return f"{city}, {state}"

def geocode(query):
    try:
        r = requests.get(NOMINATIM_URL, params={'q': query, 'format': 'json', 'limit': 1},
                         headers=HEADERS, timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        return {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon']),
                'display_name': data[0]['display_name']}
    except Exception:
        return None

# STEP 1 — load + normalize analysis data
print(f"Loading {ANALYSIS_FILE}...")
df = pd.read_csv(ANALYSIS_FILE, low_memory=False)
print(f"  {len(df)} rows")

birth_city_col, birth_state_col = 'birth_city', 'birth_state'
hs_city_col  = 'hs_city'  if 'hs_city'  in df.columns else 'school_city'
hs_state_col = 'hs_state' if 'hs_state' in df.columns else 'school_state'

fix_counts = {'placeholder_dropped': 0, 'typo_or_case_fixed': 0}
for ccol, scol in [(birth_city_col, birth_state_col), (hs_city_col, hs_state_col)]:
    if ccol not in df.columns or scol not in df.columns:
        continue
    new_cities, new_states = [], []
    for c, s in zip(df[ccol], df[scol]):
        fixed_c, fixed_s = apply_fixes(c, s)
        if fixed_c is None:
            fix_counts['placeholder_dropped'] += 1
            new_cities.append(None); new_states.append(None)
            continue
        # count a fix when EITHER the city or the state changed. (This
        # previously compared only the state, so city-only corrections --
        # which is most of them -- were never counted and the tally always
        # read 0 even when fixes were being applied.)
        city_changed = pd.notna(c) and str(c).strip() != str(fixed_c)
        state_changed = pd.notna(s) and str(s).strip() != str(fixed_s)
        if city_changed or state_changed:
            fix_counts['typo_or_case_fixed'] += 1
        new_cities.append(fixed_c); new_states.append(fixed_s)
    df[ccol] = new_cities
    df[scol] = new_states

print(f"  Fixes applied: {fix_counts}")
df.to_csv(ANALYSIS_FILE, index=False, encoding='utf-8-sig')
print(f"  Wrote cleaned {ANALYSIS_FILE}")

# STEP 2 — collect needed pairs
needed = set()
for ccol, scol in [(birth_city_col, birth_state_col), (hs_city_col, hs_state_col)]:
    if ccol not in df.columns or scol not in df.columns:
        continue
    for c, s in zip(df[ccol], df[scol]):
        k = norm_key(c, s)
        if k:
            needed.add(k)
print(f"\nUnique (city,state) pairs needed: {len(needed)}")

# STEP 3 — load cache + selectively retry failures
#
# Previously this evicted EVERY failed entry on every run, so genuinely
# unresolvable towns (bad spellings, defunct names, places Nominatim lacks)
# were re-queried at 1 req/sec on every single pipeline run, forever.
# Now each failure records an attempt count and we only retry entries that
# have not yet exhausted MAX_RETRY_ATTEMPTS. Set FORCE_RETRY_ALL = True to
# clear that and retry everything once (e.g. after fixing the typo table).
MAX_RETRY_ATTEMPTS = 2
FORCE_RETRY_ALL = False

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
print(f"Cache entries: {len(cache)}")

# Seed hand-verified coordinates for places Nominatim cannot resolve.
# These overwrite any failed entry and are never re-queried.
seeded = 0
for (city, state), (mlat, mlon) in MANUAL_COORDS.items():
    key = f"{city}|{state}"
    if cache.get(key, {}).get('lat') is None:
        cache[key] = {'lat': mlat, 'lon': mlon, 'status': 'OK',
                      'display_name': f"{city}, {state} (manual)",
                      'source': 'manual'}
        seeded += 1
if seeded:
    print(f"Seeded {seeded} hand-verified coordinate(s) from MANUAL_COORDS")

prior_attempts = {k: int(v.get('attempts', 1))
                  for k, v in cache.items()
                  if v.get('lat') is None or v.get('lon') is None}
evicted = 0
retired = 0
for k in list(cache.keys()):
    entry = cache[k]
    if entry.get('lat') is not None and entry.get('lon') is not None:
        continue  # good entry, keep
    attempts = int(entry.get('attempts', 1))
    if FORCE_RETRY_ALL or attempts < MAX_RETRY_ATTEMPTS:
        del cache[k]
        evicted += 1
    else:
        retired += 1
print(f"Retrying {evicted} failed entries; "
      f"{retired} permanently retired after {MAX_RETRY_ATTEMPTS} failed attempts")
if retired:
    print("  (set FORCE_RETRY_ALL = True in this script to retry those too)")

# STEP 4 — re-geocode missing
to_geocode = [k for k in needed if k not in cache]
print(f"\nPairs to geocode: {len(to_geocode)}")
if to_geocode:
    print(f"  Estimated time: ~{len(to_geocode)/60:.1f} minutes at 1 req/sec\n")

ok_count = fail_count = skip_count = 0
for i, key in enumerate(to_geocode, start=1):
    city, state = key.split('|', 1)
    query = build_query(city, state)
    prior = int(prior_attempts.get(key, 0))
    if not query:
        cache[key] = {'lat': None, 'lon': None, 'display_name': None,
                      'status': 'no_query', 'query': None, 'attempts': prior + 1}
        skip_count += 1
        continue
    result = geocode(query)
    if result:
        result['status'] = 'OK'; result['query'] = query
        cache[key] = result
        ok_count += 1
    else:
        cache[key] = {'lat': None, 'lon': None, 'display_name': None,
                      'status': 'failed', 'query': query, 'attempts': prior + 1}
        fail_count += 1
    time.sleep(1.0)
    if i % 50 == 0:
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        print(f"  Progress: {i}/{len(to_geocode)} (ok={ok_count}, failed={fail_count}, skipped={skip_count})")

with open(CACHE_FILE, 'w') as f:
    json.dump(cache, f)
print(f"\nGeocoding done. ok={ok_count}, failed={fail_count}, skipped={skip_count}")

# STEP 5 — rebuild coords file
rows = []
for key, val in cache.items():
    if '|' not in key:
        continue
    city, state = key.split('|', 1)
    rows.append({'city': city, 'state': state, 'lat': val.get('lat'), 'lon': val.get('lon'),
                 'status': val.get('status', 'OK' if val.get('lat') is not None else 'failed'),
                 'display_name': val.get('display_name')})
coords_df = pd.DataFrame(rows)
coords_df.to_csv(COORDS_FILE, index=False, encoding='utf-8-sig')
n_ok = (coords_df['lat'].notna() & coords_df['lon'].notna()).sum()
print(f"\nRebuilt {COORDS_FILE}: {len(coords_df)} entries, {n_ok} with lat/lon ({n_ok/len(coords_df)*100:.1f}%)")

# STEP 6 — report
good_keys = set(coords_df.loc[coords_df['lat'].notna() & coords_df['lon'].notna()]
                .apply(lambda r: f"{r['city']}|{r['state']}", axis=1))
still_missing = needed - good_keys
with open(REPORT_FILE, 'w') as f:
    f.write("Fix-and-Refill Report\n" + "="*60 + "\n\n")
    f.write(f"Data fixes applied: {fix_counts}\n\n")
    f.write(f"Pairs needed:          {len(needed)}\n")
    f.write(f"Successfully geocoded: {len(needed & good_keys)} ({len(needed & good_keys)/len(needed)*100:.1f}%)\n")
    f.write(f"Still missing:         {len(still_missing)}\n\n")
    if still_missing:
        f.write("Sample still-missing (up to 50):\n")
        for k in list(still_missing)[:50]:
            f.write(f"  {k}\n")
print(f"\nFinal coverage: {len(needed & good_keys)}/{len(needed)} ({len(needed & good_keys)/len(needed)*100:.1f}%)")
print(f"Saved: {REPORT_FILE}")
