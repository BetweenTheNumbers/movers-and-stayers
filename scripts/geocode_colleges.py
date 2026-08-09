"""
Geocode the distinct COLLEGE cities from the resolved crosswalk + per-player
overrides, reusing the SAME cache and query logic as v3_geocode.py. Only cities
not already cached (from the birth/HS work) get a Nominatim call.

Sources of college cities:
  1. college_locations_resolved.csv  (school -> city,state)
  2. PLAYER_OVERRIDE cities in college_crosswalk.py (the concordia splits)

Cache key format "City|State" and value shape match v3_geocode.py exactly, so
downstream distance code reads them identically.

Run:  python scripts/geocode_colleges.py
"""
import os
import sys
import json
import time
import pandas as pd
import requests

from college_crosswalk import PLAYER_OVERRIDE

CACHE_FILE = 'geocode_cache_v3.json'
NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
HEADERS = {'User-Agent': 'DraftMobilityResearch/1.0 (academic research)'}
RESOLVED = 'college_locations_resolved.csv'

COUNTRY_MAP = {
    'PR': 'Puerto Rico', 'DR': 'Dominican Republic', 'VE': 'Venezuela',
    'MX': 'Mexico', 'CU': 'Cuba', 'JA': 'Japan', 'JP': 'Japan',
    'KO': 'South Korea', 'AU': 'Australia', 'AS': 'American Samoa',
    'VI': 'US Virgin Islands', 'GU': 'Guam',
}
US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID',
    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS',
    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK',
    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV',
    'WI', 'WY', 'DC',
}
CA_PROVS = {'ON', 'QC', 'BC', 'AB', 'MB', 'SK', 'NS', 'NB', 'NL', 'PE',
            'YT', 'NT', 'NU'}


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def build_query(city, state):
    if not city or not state:
        return None
    city = str(city).strip()
    state = str(state).strip()
    if state in US_STATES:
        return f"{city}, {state}, USA"
    elif state in CA_PROVS:
        return f"{city}, {state}, Canada"
    elif state in COUNTRY_MAP:
        return f"{city}, {COUNTRY_MAP[state]}"
    else:
        return f"{city}, {state}"


def geocode(query):
    try:
        r = requests.get(NOMINATIM_URL,
                         params={'q': query, 'format': 'json', 'limit': 1},
                         headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        return {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon']),
                'display_name': data[0]['display_name']}
    except Exception:
        return None


def main():
    if not os.path.exists(RESOLVED):
        print(f"ERROR: {RESOLVED} not found. Run resolve_colleges.py first.")
        sys.exit(1)

    # collect distinct college cities
    cities = set()
    res = pd.read_csv(RESOLVED)
    for _, r in res.iterrows():
        c, s = str(r.get('city', '')).strip(), str(r.get('state', '')).strip()
        if c and s and c.lower() != 'nan':
            cities.add((c, s))
    # add per-player override cities
    for city, state in PLAYER_OVERRIDE.values():
        cities.add((str(city).strip(), str(state).strip()))

    print(f"Distinct college cities: {len(cities)}")

    cache = load_cache()
    print(f"Already cached (from birth/HS work): {len(cache)}")
    to_geocode = [c for c in cities if f"{c[0]}|{c[1]}" not in cache]
    print(f"NEW college cities to geocode: {len(to_geocode)}")
    if to_geocode:
        print(f"Estimated time: {len(to_geocode)/60:.1f} min at 1 req/sec\n")
    else:
        print("Nothing new to geocode -- all college cities already cached.\n")

    new_ok = 0
    fails = 0
    for i, (city, state) in enumerate(to_geocode):
        key = f"{city}|{state}"
        query = build_query(city, state)
        if not query:
            cache[key] = {'lat': None, 'lon': None, 'display_name': None,
                          'status': 'no_query', 'attempts': 1}
            continue
        result = geocode(query)
        if result:
            result['status'] = 'OK'
            cache[key] = result
            new_ok += 1
        else:
            cache[key] = {'lat': None, 'lon': None, 'display_name': None,
                          'status': 'failed', 'query': query, 'attempts': 1}
            fails += 1
        time.sleep(1.0)
        if (i + 1) % 50 == 0:
            save_cache(cache)
            print(f"  {i+1}/{len(to_geocode)} ({new_ok} ok, {fails} failed)")

    save_cache(cache)
    print(f"\nDone. New geocoded: {new_ok}, failed: {fails}")
    print(f"Cache now has {len(cache)} entries.")
    if fails:
        print("Failed cities are cached with status='failed'; re-runnable later.")


if __name__ == '__main__':
    main()
