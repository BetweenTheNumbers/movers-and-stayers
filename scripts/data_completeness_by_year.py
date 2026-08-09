"""
Data completeness audit by draft year (utility, not a pipeline step).

Answers: how far back can this analysis credibly go? Reports, for every draft
year in the register, the fields the mover analysis actually depends on:

  - birth place parsed          (needed for the birth coordinate)
  - HS "(City,ST)" parsed       (needed for the school coordinate)
  - both present                (the binding constraint)
  - HS draftees identified      (playerClass / schoolDivision populated)
  - mlbid present               (needed for the FanGraphs WAR join)
  - draft phase mix             (pre-1987 had January and secondary phases,
                                 which draw from a different population)

Usage:
    python scripts/data_completeness_by_year.py
    python scripts/data_completeness_by_year.py --csv
    python scripts/data_completeness_by_year.py --from 1965 --to 2025

Output:
    data_completeness_by_year.csv   (with --csv)
"""

import os
import re
import sys
import pandas as pd

REGISTER = 'data/tbc_draft_register.csv'
OUT_CSV = 'data_completeness_by_year.csv'


def arg(flag, default):
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def has_place(p):
    return bool(p) and not pd.isna(p) and ',' in str(p)


def has_school_city(s):
    if pd.isna(s):
        return False
    return bool(re.search(r'\(([^()]+,[^()]+)\)\s*$', str(s)))


def main():
    if not os.path.exists(REGISTER):
        print(f"ERROR: {REGISTER} not found. Run from the project root.")
        sys.exit(1)

    y0, y1 = arg('--from', 1965), arg('--to', 2025)
    df = pd.read_csv(REGISTER, low_memory=False)
    df = df[(df['year'] >= y0) & (df['year'] <= y1)].copy()
    print(f"\nRegister rows {y0}-{y1}: {len(df):,}\n")

    df['_birth'] = df['place'].apply(has_place)
    df['_hs'] = df['school'].apply(has_school_city)
    df['_both'] = df['_birth'] & df['_hs']
    df['_ishs'] = ((df.get('playerClass') == 'HS') |
                   (df.get('schoolDivision') == 'HS'))
    if 'mlbid' in df.columns:
        df['_mlbid'] = df['mlbid'].fillna(0).astype(str).str.strip().ne('0')
    else:
        df['_mlbid'] = False
    df['_mlb'] = df['highLevel'].eq('MLB') if 'highLevel' in df.columns else False

    phase_col = 'phase' if 'phase' in df.columns else None

    rows = []
    for yr, g in df.groupby('year'):
        n = len(g)
        hs = g[g['_ishs']]
        rec = {
            'year': int(yr),
            'rows': n,
            'birth_pct': round(g['_birth'].mean() * 100, 1),
            'hs_city_pct': round(g['_hs'].mean() * 100, 1),
            'both_pct': round(g['_both'].mean() * 100, 1),
            'hs_draftees': len(hs),
            'hs_both_pct': round(hs['_both'].mean() * 100, 1) if len(hs) else 0.0,
            'mlbid_pct': round(g['_mlbid'].mean() * 100, 1),
            'reached_mlb_pct': round(g['_mlb'].mean() * 100, 1),
        }
        if phase_col:
            top = g[phase_col].astype(str).value_counts()
            rec['main_phase'] = top.index[0] if len(top) else ''
            rec['n_phases'] = int(g[phase_col].nunique())
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values('year')

    print(f"{'Year':>5} {'Rows':>6} {'Birth%':>7} {'HSCity%':>8} {'Both%':>7} "
          f"{'HSdraft':>8} {'HSboth%':>8} {'mlbid%':>7} {'MLB%':>6}"
          + ("  Phase" if phase_col else ""))
    print("-" * (78 + (10 if phase_col else 0)))
    for _, r in out.iterrows():
        line = (f"{int(r['year']):>5} {int(r['rows']):>6,} {r['birth_pct']:>6.1f}% "
                f"{r['hs_city_pct']:>7.1f}% {r['both_pct']:>6.1f}% "
                f"{int(r['hs_draftees']):>8,} {r['hs_both_pct']:>7.1f}% "
                f"{r['mlbid_pct']:>6.1f}% {r['reached_mlb_pct']:>5.1f}%")
        if phase_col:
            line += f"  {r.get('main_phase', '')} ({int(r.get('n_phases', 1))})"
        print(line)

    # era summary
    print(f"\n{'=' * 78}")
    print("ERA SUMMARY (the binding constraint is 'HS draftees with both locations')")
    print("=" * 78)
    eras = [(1965, 1979), (1980, 1989), (1990, 1995), (1996, 2005),
            (2006, 2019), (2020, 2025)]
    print(f"{'Era':<12} {'Rows':>8} {'HS draftees':>12} {'HS w/ both':>12} {'usable%':>9}")
    print("-" * 78)
    for lo, hi in eras:
        g = df[(df['year'] >= lo) & (df['year'] <= hi)]
        if g.empty:
            continue
        hs = g[g['_ishs']]
        usable = int(hs['_both'].sum())
        pct = usable / len(hs) * 100 if len(hs) else 0
        print(f"{f'{lo}-{hi}':<12} {len(g):>8,} {len(hs):>12,} {usable:>12,} {pct:>8.1f}%")

    if phase_col:
        print(f"\nDRAFT PHASES BY ERA (pre-1987 had January and secondary phases,")
        print("which draw from a different population and are not comparable):")
        for lo, hi in eras:
            g = df[(df['year'] >= lo) & (df['year'] <= hi)]
            if g.empty:
                continue
            vc = g[phase_col].astype(str).value_counts()
            summary = ', '.join(f"{k}={v:,}" for k, v in vc.head(4).items())
            print(f"  {lo}-{hi}: {summary}")

    if '--csv' in sys.argv:
        out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
        print(f"\nSaved: {OUT_CSV}")
    else:
        print(f"\n(Use --csv to write {OUT_CSV})")
    print()


if __name__ == '__main__':
    main()
