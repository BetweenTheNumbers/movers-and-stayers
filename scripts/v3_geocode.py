"""
Step 2 — Geocode all unique birth + school cities from v3_analysis.csv.
Uses Nominatim (free, 1 req/sec, no API key). Caches aggressively.

Output: city_coords_v3.csv + geocode_cache_v3.json
"""

import pandas as pd
import requests
import json
import os
import time

CACHE_FILE = 'geocode_cache_v3.json'
NOMINATIM_URL = 'https://nominatim.openstreetmap.org/search'
HEADERS = {'User-Agent': 'DraftMobilityResearch/1.0 (academic research)'}

COUNTRY_MAP = {
    'PR': 'Puerto Rico', 'DR': 'Dominican Republic', 'VE': 'Venezuela',
    'MX': 'Mexico', 'CU': 'Cuba', 'JA': 'Japan', 'JP': 'Japan',
    'KO': 'South Korea', 'AU': 'Australia', 'AS': 'American Samoa',
    'VI': 'US Virgin Islands', 'GU': 'Guam',
}

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}
CA_PROVS = {'ON','QC','BC','AB','MB','SK','NS','NB','NL','PE','YT','NT','NU'}

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
    city = city.strip()
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
        r = requests.get(NOMINATIM_URL, params={'q': query, 'format': 'json', 'limit': 1},
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
    df = pd.read_csv('v3_analysis.csv', low_memory=False)
    print(f"Loaded {len(df)} rows")

    cities = set()
    for _, r in df.iterrows():
        if pd.notna(r.get('birth_city')) and pd.notna(r.get('birth_state')):
            cities.add((str(r['birth_city']).strip(), str(r['birth_state']).strip()))
        if pd.notna(r.get('school_city')) and pd.notna(r.get('school_state')):
            cities.add((str(r['school_city']).strip(), str(r['school_state']).strip()))
    print(f"Unique (city, state) pairs to geocode: {len(cities)}")

    cache = load_cache()
    print(f"Already cached: {len(cache)}")
    to_geocode = [c for c in cities if f"{c[0]}|{c[1]}" not in cache]
    print(f"Need to geocode: {len(to_geocode)}")
    if to_geocode:
        print(f"Estimated time: {len(to_geocode)/60:.1f} minutes at 1 req/sec\n")

    new_geocodes = 0
    failures = 0
    for i, (city, state) in enumerate(to_geocode):
        key = f"{city}|{state}"
        query = build_query(city, state)
        if not query:
            cache[key] = {'lat': None, 'lon': None, 'display_name': None, 'status': 'no_query', 'attempts': 1}
            continue
        result = geocode(query)
        if result:
            result['status'] = 'OK'
            cache[key] = result
            new_geocodes += 1
        else:
            cache[key] = {'lat': None, 'lon': None, 'display_name': None, 'status': 'failed', 'query': query, 'attempts': 1}
            failures += 1
        time.sleep(1.0)
        if (i + 1) % 100 == 0:
            save_cache(cache)
            print(f"  Progress: {i+1}/{len(to_geocode)} ({new_geocodes} ok, {failures} failed)")

    save_cache(cache)
    print(f"\nTotal cached: {len(cache)}")
    print(f"New geocodes: {new_geocodes}, failures: {failures}")

    rows = []
    for key, val in cache.items():
        if '|' not in key:
            continue
        city, state = key.split('|', 1)
        rows.append({'city': city, 'state': state, 'lat': val.get('lat'),
                     'lon': val.get('lon'), 'status': val.get('status'),
                     'display_name': val.get('display_name')})
    out = pd.DataFrame(rows)
    out.to_csv('city_coords_v3.csv', index=False, encoding='utf-8-sig')
    print(f"\nGeocoding summary:")
    print(f"  Total entries:  {len(out)}")
    print(f"  Successful:     {(out['status']=='OK').sum()} ({(out['status']=='OK').mean()*100:.1f}%)")
    print(f"  Failed:         {(out['status']!='OK').sum()}")
    print(f"\nSaved: city_coords_v3.csv")

    failed = out[out['status']!='OK']
    if len(failed) > 0:
        print(f"\nSample failed lookups:")
        print(failed.head(15).to_string())

if __name__ == '__main__':
    main()
