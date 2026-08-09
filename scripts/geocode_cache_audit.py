"""
Geocode cache audit (utility, not a pipeline step).

Reports the health of geocode_cache_v3.json so you can see at a glance how
many places are resolved, how many failed, and how many are still eligible
for retry. Also lets you retire or reset failures without hand-editing JSON.

Usage:
    python scripts/geocode_cache_audit.py              # report only
    python scripts/geocode_cache_audit.py --list 40    # show the worst offenders
    python scripts/geocode_cache_audit.py --export     # write geocode_failures.csv
    python scripts/geocode_cache_audit.py --import     # read manual lat/lon back in
    python scripts/geocode_cache_audit.py --retire     # stop retrying all failures
    python scripts/geocode_cache_audit.py --reset      # make all failures retryable again

Manual lookup workflow:
    1. python scripts/geocode_cache_audit.py --export
    2. Open geocode_failures.csv. Each row has a ready-made search URL.
       Look the place up, paste the coordinates into the lat and lon columns.
       Leave rows you cannot resolve blank; they are simply skipped.
    3. python scripts/geocode_cache_audit.py --import
       Filled rows are merged into the cache as resolved entries and will
       never be re-queried. Save the CSV first: an open file blocks the read.

Background: step 3 (fix_and_refill_geocodes.py) used to evict every failed
entry on every run, so unresolvable towns were re-queried at 1 req/sec
forever. Failures now carry an `attempts` counter and are retired after
MAX_RETRY_ATTEMPTS. This tool inspects and adjusts that state.
"""

import csv
import json
import os
import sys
import urllib.parse
from collections import Counter

CACHE_FILE = 'geocode_cache_v3.json'
FAILURES_CSV = 'geocode_failures.csv'
RETIRE_AT = 99  # attempts value that means "never retry again"


def load():
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run from the project root.")
        sys.exit(1)
    with open(CACHE_FILE, encoding='utf-8') as f:
        return json.load(f)


def save(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f)


def is_good(v):
    return v.get('lat') is not None and v.get('lon') is not None


def export_failures(bad):
    """Write failures to CSV with a search URL so they can be looked up by hand."""
    rows = []
    for key, v in sorted(bad.items()):
        city, state = (key.split('|', 1) + [''])[:2]
        q = v.get('query') or f"{city}, {state}"
        url = "https://www.openstreetmap.org/search?query=" + urllib.parse.quote(q)
        rows.append({
            'city': city,
            'state': state,
            'attempts': v.get('attempts', 1),
            'status': v.get('status', 'failed'),
            'query_tried': q,
            'search_url': url,
            'lat': '',        # <- fill these two in by hand
            'lon': '',
            'notes': '',
        })
    with open(FAILURES_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ['city', 'state', 'attempts', 'status', 'query_tried',
                            'search_url', 'lat', 'lon', 'notes'])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Wrote {len(rows):,} failures to {FAILURES_CSV}")
    print("  Fill in the lat and lon columns, save, then run with --import.")


def import_failures(cache):
    """Merge manually entered lat/lon back into the cache."""
    if not os.path.exists(FAILURES_CSV):
        print(f"\n  ERROR: {FAILURES_CSV} not found. Run with --export first.")
        return
    try:
        with open(FAILURES_CSV, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
    except PermissionError:
        print(f"\n  ERROR: {FAILURES_CSV} is open in another program (Excel?). "
              f"Close it and retry.")
        return

    merged = skipped = malformed = 0
    for r in rows:
        lat_s, lon_s = (r.get('lat') or '').strip(), (r.get('lon') or '').strip()
        if not lat_s or not lon_s:
            skipped += 1
            continue
        try:
            lat, lon = float(lat_s), float(lon_s)
        except ValueError:
            print(f"    skipping {r.get('city')},{r.get('state')}: "
                  f"could not read lat/lon ('{lat_s}', '{lon_s}')")
            malformed += 1
            continue
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            print(f"    skipping {r.get('city')},{r.get('state')}: "
                  f"coordinates out of range ({lat}, {lon})")
            malformed += 1
            continue
        key = f"{(r.get('city') or '').strip()}|{(r.get('state') or '').strip()}"
        cache[key] = {'lat': lat, 'lon': lon, 'status': 'OK',
                      'display_name': (r.get('notes') or '').strip() or None,
                      'source': 'manual'}
        merged += 1

    save(cache)
    print(f"\n  Merged {merged:,} manually resolved places into the cache.")
    if skipped:
        print(f"  Left {skipped:,} rows blank (still unresolved).")
    if malformed:
        print(f"  Rejected {malformed:,} rows with unreadable coordinates.")
    if merged:
        print("  These are now permanent and will never be re-queried.")
        print("  Re-run step 5 to pick them up: python run_pipeline.py --only 5")


def main():
    args = sys.argv[1:]
    cache = load()

    good = {k: v for k, v in cache.items() if is_good(v)}
    bad = {k: v for k, v in cache.items() if not is_good(v)}

    # Action modes run first and return, so the report never shows stale counts.
    if '--export' in args:
        print(f"\nGeocode cache: {CACHE_FILE} ({len(cache):,} entries, "
              f"{len(bad):,} failed)")
        if not bad:
            print("  No failures to export.")
        else:
            export_failures(bad)
        print()
        return

    if '--import' in args:
        print(f"\nGeocode cache: {CACHE_FILE} ({len(cache):,} entries, "
              f"{len(bad):,} failed before import)")
        import_failures(cache)
        print()
        return

    print(f"\nGeocode cache: {CACHE_FILE}")
    print(f"  Total entries:   {len(cache):,}")
    print(f"  Resolved:        {len(good):,} ({len(good)/max(len(cache),1)*100:.1f}%)")
    print(f"  Failed:          {len(bad):,}")

    if bad:
        att = Counter(int(v.get('attempts', 1)) for v in bad.values())
        print("\n  Failures by attempt count:")
        for a in sorted(att):
            label = ' (retired)' if a >= RETIRE_AT else ''
            print(f"    {a} attempt(s): {att[a]:,}{label}")
        statuses = Counter(v.get('status', 'unknown') for v in bad.values())
        print("\n  Failure reasons:")
        for s, n in statuses.most_common():
            print(f"    {s}: {n:,}")

    if '--list' in args:
        try:
            n = int(args[args.index('--list') + 1])
        except (IndexError, ValueError):
            n = 25
        print(f"\n  First {n} failed places:")
        for k in list(bad)[:n]:
            v = bad[k]
            print(f"    {k:<40} attempts={v.get('attempts', 1)} "
                  f"status={v.get('status', '?')}")

    if '--retire' in args:
        for k in bad:
            cache[k]['attempts'] = RETIRE_AT
        save(cache)
        print(f"\n  Retired {len(bad):,} failures. They will not be retried again.")
        print("  (Run with --reset to undo, or set FORCE_RETRY_ALL in step 3.)")

    elif '--reset' in args:
        for k in bad:
            cache[k]['attempts'] = 0
        save(cache)
        print(f"\n  Reset {len(bad):,} failures to retryable.")

    if not any(a in args for a in ('--retire', '--reset')):
        retryable = sum(1 for v in bad.values() if int(v.get('attempts', 1)) < 2)
        if retryable:
            print(f"\n  {retryable:,} failures are still eligible for retry on the next "
                  f"step-3 run.\n  Use --retire to stop re-querying them entirely.")
        else:
            print("\n  No failures are eligible for retry. Nothing will be re-queried.")
    print()


if __name__ == '__main__':
    main()
