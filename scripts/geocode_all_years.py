"""
Geocode every draft year (utility, not a pipeline step).

The main pipeline only geocodes the configured cohort window. This widens
that to the whole register so historical eras can be analyzed, writing into
the SAME geocode_cache_v3.json the pipeline uses. Nothing already cached is
re-queried, so it is safe to run repeatedly and safe to interrupt.

Uses the shared tables in geo_tables.py, so typo corrections and country
mappings match the rest of the pipeline exactly.

Usage:
    python scripts/geocode_all_years.py                 # 1965-2025
    python scripts/geocode_all_years.py --from 1987     # narrower range
    python scripts/geocode_all_years.py --dry-run       # count only, no network
    python scripts/geocode_all_years.py --limit 500     # cap this session

Safe to stop with Ctrl+C: the cache is written every 25 lookups and on exit.
"""

import json
import os
import re
import signal
import sys
import time

import pandas as pd

from geo_tables import (US_STATES, CA_PROVS, COUNTRY_MAP, MANUAL_COORDS,
                        build_query, apply_typo_fix)

REGISTER = 'data/tbc_draft_register.csv'
BONUS = 'data/tbc_signing_bonus.csv'
CACHE_FILE = 'geocode_cache_v3.json'
SAVE_EVERY = 25
SLEEP = 1.0          # Nominatim policy: 1 request/second. Do not lower.
MAX_RETRY_ATTEMPTS = 2

_interrupted = False


def _on_sigint(signum, frame):
    global _interrupted
    _interrupted = True
    print("\n  Interrupt received — saving cache and stopping cleanly...")


signal.signal(signal.SIGINT, _on_sigint)


def arg(flag, default, cast=int):
    if flag in sys.argv:
        try:
            return cast(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def parse_place(p):
    if pd.isna(p) or ',' not in str(p):
        return None, None
    a, b = str(p).rsplit(',', 1)
    return a.strip(), b.strip()


def parse_school_city(s):
    if pd.isna(s):
        return None, None
    m = re.search(r'\(([^()]+)\)\s*$', str(s))
    if not m or ',' not in m.group(1):
        return None, None
    a, b = m.group(1).rsplit(',', 1)
    return a.strip(), b.strip()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    tmp = CACHE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(cache, f)
    os.replace(tmp, CACHE_FILE)   # atomic, so an interrupt cannot corrupt it


def geocode_one(query):
    try:
        import requests
        r = requests.get('https://nominatim.openstreetmap.org/search',
                         params={'q': query, 'format': 'json', 'limit': 1},
                         headers={'User-Agent': 'DraftMobilityResearch/1.0'},
                         timeout=15)
        if r.status_code == 200:
            j = r.json()
            if j:
                return {'lat': float(j[0]['lat']), 'lon': float(j[0]['lon']),
                        'display_name': j[0].get('display_name'), 'status': 'OK'}
        elif r.status_code == 429:
            print("    rate limited; pausing 30s")
            time.sleep(30)
    except Exception:
        return None
    return None


def collect_places(y0, y1):
    if not os.path.exists(REGISTER):
        print(f"ERROR: {REGISTER} not found. Run from the project root.")
        sys.exit(1)
    reg = pd.read_csv(REGISTER, low_memory=False)
    reg = reg[(reg['year'] >= y0) & (reg['year'] <= y1)]
    print(f"Register rows {y0}-{y1}: {len(reg):,}")

    places = set()
    for p in reg['place'].dropna().unique():
        c, s = parse_place(p)
        if c and s:
            places.add(apply_typo_fix(c, s))
    for sch in reg['school'].dropna().unique():
        c, s = parse_school_city(sch)
        if c and s:
            places.add(apply_typo_fix(c, s))

    if os.path.exists(BONUS):
        sb = pd.read_csv(BONUS, low_memory=False)
        col = 'hsplace' if 'hsplace' in sb.columns else None
        if col:
            for p in sb[col].dropna().unique():
                c, s = parse_place(p)
                if c and s:
                    places.add(apply_typo_fix(c, s))
    return places


def main():
    y0, y1 = arg('--from', 1965), arg('--to', 2025)
    limit = arg('--limit', None)
    dry = '--dry-run' in sys.argv

    places = collect_places(y0, y1)
    print(f"Unique (city, state) pairs: {len(places):,}")

    cache = load_cache()
    print(f"Cache entries: {len(cache):,}")

    # seed hand-verified coordinates
    seeded = 0
    for (city, state), (mlat, mlon) in MANUAL_COORDS.items():
        key = f"{city}|{state}"
        if cache.get(key, {}).get('lat') is None:
            cache[key] = {'lat': mlat, 'lon': mlon, 'status': 'OK',
                          'display_name': f"{city}, {state} (manual)",
                          'source': 'manual'}
            seeded += 1
    if seeded:
        print(f"Seeded {seeded} manual coordinate(s)")

    todo = []
    for (c, s) in sorted(places):
        key = f"{c}|{s}"
        e = cache.get(key)
        if e is None:
            todo.append((c, s))
        elif e.get('lat') is None and int(e.get('attempts', 1)) < MAX_RETRY_ATTEMPTS:
            todo.append((c, s))

    print(f"Need geocoding: {len(todo):,}")
    if limit:
        todo = todo[:limit]
        print(f"  (limited to {len(todo):,} this session)")
    if not todo:
        save_cache(cache)
        print("Nothing to do. Cache is complete for this range.\n")
        return
    mins = len(todo) * SLEEP / 60
    print(f"Estimated time: {mins:.0f} min ({mins/60:.1f} h) at {SLEEP:.0f} req/sec\n")

    if dry:
        pd.DataFrame(todo, columns=['city', 'state']).to_csv(
            'geocode_todo_preview.csv', index=False, encoding='utf-8-sig')
        print("Dry run — wrote geocode_todo_preview.csv, no network calls made.\n")
        return

    ok = fail = 0
    start = time.time()
    for i, (c, s) in enumerate(todo, 1):
        if _interrupted:
            break
        key = f"{c}|{s}"
        q = build_query(c, s)
        if not q:
            cache[key] = {'lat': None, 'lon': None, 'status': 'no_query',
                          'attempts': int(cache.get(key, {}).get('attempts', 0)) + 1}
            continue
        res = geocode_one(q)
        if res:
            cache[key] = res
            ok += 1
        else:
            prior = int(cache.get(key, {}).get('attempts', 0))
            cache[key] = {'lat': None, 'lon': None, 'status': 'failed',
                          'query': q, 'attempts': prior + 1}
            fail += 1
        time.sleep(SLEEP)
        if i % SAVE_EVERY == 0:
            save_cache(cache)
            el = time.time() - start
            rate = i / el if el else 0
            left = (len(todo) - i) / rate / 60 if rate else 0
            print(f"  {i:,}/{len(todo):,}  ok={ok:,} fail={fail:,}  "
                  f"~{left:.0f} min left")

    save_cache(cache)
    print(f"\nDone. ok={ok:,} failed={fail:,}"
          + ("  (interrupted early)" if _interrupted else ""))
    resolved = sum(1 for v in cache.values() if v.get('lat') is not None)
    print(f"Cache now: {len(cache):,} entries, {resolved:,} resolved "
          f"({resolved/len(cache)*100:.1f}%)")
    print("\nNext: python scripts/geocode_cache_audit.py --export "
          "to review any failures.\n")


if __name__ == '__main__':
    main()
