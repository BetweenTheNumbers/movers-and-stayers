"""
Verify what each FanGraphs file actually CONTAINS vs what its name claims.

Prints, for every fg* CSV, whether the columns look like hitting, pitching,
or fielding — so we can see if any filename is lying about its content.

Run:  python scripts/verify_fg_files.py
"""

import glob
import pandas as pd


def kind(cols):
    cl = {c.lower() for c in cols}
    # pitching-only signals
    if {'era', 'fip', 'xfip', 'k/9', 'bb/9'} & cl:
        return 'PITCHING'
    # fielding-only signals
    if {'uzr', 'drs', 'oaa', 'frv', 'inn', 'rngr'} & cl:
        return 'FIELDING'
    # hitting signals
    if {'woba', 'wrc+', 'obp', 'slg', 'pa', 'iso'} & cl:
        return 'HITTING'
    return 'UNKNOWN'


def claims(name):
    n = name.lower()
    if 'hit' in n:
        return 'HITTING'
    if 'pit' in n:
        return 'PITCHING'
    if 'fld' in n:
        return 'FIELDING'
    return '?'


print(f"{'file':<26} {'name claims':<12} {'content is':<12} {'match?'}")
print('-' * 60)
for f in sorted(glob.glob('fg*.csv')):
    try:
        d = pd.read_csv(f, nrows=5)
    except Exception as e:
        print(f'{f:<26} (could not read: {e})')
        continue
    c, k = claims(f), kind(d.columns)
    ok = 'OK' if c == k else '*** MISMATCH ***'
    print(f'{f:<26} {c:<12} {k:<12} {ok}')
print()
print("Every row should say OK. Any MISMATCH means the filename lies about")
print("its content and must be renamed (or the script pointed the other way).")
