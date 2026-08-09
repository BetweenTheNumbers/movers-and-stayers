"""
Determine how traded-player seasons are represented in the FanGraphs season
files, so the join filters them correctly (and never double-counts).

Set FILES below to match what is actually on disk. By CONTENT:
  - the hitting file has PA / AVG / wOBA / wRC+
  - the pitching file has W / L / SV / IP / ERA / FIP
If you have renamed the files, just point these at the right paths.

Run:  python scripts/check_totals.py
"""

import os
import sys
import pandas as pd

# (label, filename). Adjust filenames here if you renamed them on disk.
FILES = [
    ('HITTING',  'fg-pit-seasons.csv'),   # currently mislabeled: holds hitting
    ('PITCHING', 'fg-hit-seasons.csv'),   # currently mislabeled: holds pitching
]


def detect_kind(cols):
    cl = {c.lower() for c in cols}
    if {'era', 'fip', 'ip'} & cl:
        return 'PITCHING'
    if {'woba', 'wrc+', 'obp', 'slg', 'pa'} & cl:
        return 'HITTING'
    return 'UNKNOWN'


def main():
    for label, fname in FILES:
        print('=' * 64)
        if not os.path.exists(fname):
            print(f'{fname}: NOT FOUND (skipping)')
            continue
        d = pd.read_csv(fname, low_memory=False)
        kind = detect_kind(d.columns)
        print(f'{fname}  (labeled {label}, content looks like {kind})')
        print('  rows:', len(d))

        if 'Team' not in d.columns:
            print('  no Team column?')
            continue
        is_tot = d['Team'].astype(str).str.contains('- - -', regex=False)
        print('  "- - -" total rows:', int(is_tot.sum()))

        tot_keys = d.loc[is_tot, ['PlayerId', 'Season']].drop_duplicates()
        if len(tot_keys):
            merged = d.merge(tot_keys, on=['PlayerId', 'Season'])
            per = merged.groupby(['PlayerId', 'Season']).size()
            print('  among player-seasons that HAVE a "- - -" row:')
            print('     rows per player-season -> min', int(per.min()),
                  'max', int(per.max()))
            if per.max() == 1:
                print('     => file holds ONLY the total for traded players (CLEAN).')
                print('        Rule: use every row as-is; nothing to filter.')
            else:
                print('     => BOTH the total AND per-team splits exist.')
                print('        Rule: DROP "- - -" rows OR the team rows, not both;')
                print('        keep exactly one representation per player-season.')

        non_tot = d[~is_tot]
        dup = int(non_tot.duplicated(['PlayerId', 'Season']).sum())
        print('  dup (PlayerId,Season) among NON-total rows:', dup,
              '(0 => single-team players are unique)')
        # net rule
        full_dup = int(d.duplicated(['PlayerId', 'Season']).sum())
        print('  dup (PlayerId,Season) over the WHOLE file:', full_dup)
        if full_dup == 0:
            print('  VERDICT: already one row per player-season. Use as-is.')
        else:
            print('  VERDICT: needs de-duping to one row per player-season.')
        print()


if __name__ == '__main__':
    main()
