"""
Inventory ALL The Baseball Cube source files and every location-ish field, with fill
rates, so we can find the best route to birth-city + HS-city for college players
(not just hsplace). Read-only.

Run:  python scripts/inventory_sources.py
"""
import os
import glob
import pandas as pd

DDIR = 'data'

LOC_HINTS = ['place', 'city', 'town', 'state', 'hs', 'high', 'school', 'birth',
             'home', 'origin', 'hometown', 'residence', 'geo', 'lat', 'lon']


def main():
    files = sorted(glob.glob(os.path.join(DDIR, '*.csv')))
    if not files:
        print(f"No CSVs in {DDIR}/")
        # also check common alternates
        for alt in ['.', 'data']:
            files += glob.glob(os.path.join(alt, 'tbc*.csv'))
    print(f"Found {len(files)} source file(s) in {DDIR}/\n")

    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=200000)
        except Exception as e:
            print(f"  (couldn't read {f}: {e})")
            continue
        print("=" * 72)
        print(f"FILE: {os.path.basename(f)}   ({len(df):,} rows, {len(df.columns)} cols)")
        print("=" * 72)
        # id columns
        idcols = [c for c in df.columns if 'id' in c.lower()]
        print(f"  id-ish columns: {idcols}")
        # location-ish columns with fill rate + example
        print(f"  {'column':<22}{'fill%':>7}  example")
        for c in df.columns:
            if any(h in c.lower() for h in LOC_HINTS):
                nn = df[c].notna().sum()
                ex = ''
                if nn:
                    ex = repr(df[c].dropna().astype(str).iloc[0])[:34]
                print(f"  {c:<22}{nn/len(df)*100:>6.0f}%  {ex}")
        print()

    print("=" * 72)
    print("GOAL: find a file/field pair giving birth-city AND hs-city for college")
    print("players at HIGH coverage (>>37%). 'place' in the register = BIRTH city")
    print("(near 100%). We need an HS-city field for college players that beats")
    print("hsplace's 33-37%.")


if __name__ == '__main__':
    main()
