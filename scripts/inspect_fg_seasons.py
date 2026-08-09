"""
Inspect the FanGraphs player-season CSVs before building any rollup.

Reports, for each season file it can find: row/column counts, the full column
list, which columns look like player-id / season / team keys, and — crucially —
whether a (player, season) pair appears more than once (i.e. traded-player
stints that need combining), with a concrete example.

Run:  python scripts/inspect_fg_seasons.py
"""

import glob
import os
import pandas as pd

PATTERNS = [
    'fg-hit-seasons*', 'fg-pit-seasons*', 'fg-fld-seasons*',
    'fg_hit_seasons*', 'fg_pit_seasons*', 'fg_fld_seasons*',
    'fg-hit-season*', 'fg-pit-season*', 'fg-fld-season*',
]


def main():
    seen = set()
    found_any = False
    for pat in PATTERNS:
        for f in sorted(glob.glob(pat)):
            if f in seen:
                continue
            seen.add(f)
            found_any = True
            print('=' * 72)
            print(f, '  size =', os.path.getsize(f), 'bytes')
            d = pd.read_csv(f, low_memory=False)
            print('rows:', len(d), '  cols:', len(d.columns))
            print('columns:', list(d.columns))

            idc = [c for c in d.columns
                   if 'id' in c.lower() or c.upper() == 'MLBAMID']
            seac = [c for c in d.columns
                    if 'season' in c.lower() or 'year' in c.lower()]
            teamc = [c for c in d.columns
                     if 'team' in c.lower() or c.lower() in ('tm', 'org')]
            print('  id-ish:', idc, ' season-ish:', seac, ' team-ish:', teamc)

            if idc and seac:
                k = [idc[0], seac[0]]
                dup = int(d.duplicated(subset=k).sum())
                print(f'  duplicate ({idc[0]},{seac[0]}) rows:', dup,
                      '=> MULTI-STINT present (needs rollup)' if dup > 0
                      else '=> already one row per player-season')
                if dup > 0:
                    g = (d.groupby(k).size().reset_index(name='n')
                         .query('n > 1').head(1))
                    if len(g):
                        ex = d.merge(g[k], on=k)
                        show = [c for c in ([idc[0], seac[0]] + teamc +
                                            ['WAR', 'G', 'PA', 'IP'])
                                if c in d.columns]
                        print('  example multi-stint rows:')
                        print(ex[show].to_string(index=False))
                # flag a possible pre-existing combined/total row
                if teamc:
                    tvals = d[teamc[0]].astype(str)
                    totals = tvals[tvals.str.contains(
                        r'^-|total|TOT|2 Tms|- - -', case=False, regex=True)]
                    if len(totals):
                        print(f'  NOTE: {len(totals)} rows look like a pre-made '
                              f'"total" team row ({teamc[0]} = '
                              f'{sorted(totals.unique())[:5]}). If present, do NOT '
                              f'also sum the stints — that double-counts.')
            print()

    if not found_any:
        print("No season files matched. Files present in this directory:")
        for f in sorted(glob.glob('*.csv')):
            print("   ", f)
        print("\nTell me the exact filenames and I'll adjust the patterns.")


if __name__ == '__main__':
    main()
