"""
Market efficiency: draft VOLUME vs MLB CONVERSION by metro area.

The scouting thesis: some heavily-scouted markets (sunbelt, showcase-dense)
are OVER-drafted -- clubs spend many picks there but the conversion rate to
MLB is mediocre. Other markets may be UNDER-covered: fewer picks, but a high
hit rate on the ones taken. This surfaces both.

Two geographies are offered:
  - by BIRTH city/metro   (where talent originates)
  - by HS city/metro      (where talent is developed/seen)
Both matter; they answer different questions.

Method: grid the country into ~1-degree cells (like the heatmaps), require a
minimum draftee count, and for each cell report draft volume, MLB count, raw
conversion rate, and an empirical-Bayes-shrunk rate (so small markets don't
dominate). Cells are ranked to show the over- and under-performers.

IMPORTANT: this is DESCRIPTIVE. A low conversion rate in a high-volume market
is consistent with over-coverage, but also with other stories (deeper draft
penetration into marginal players, signability picks, etc). It flags markets
worth a look; it does not prove inefficiency.

Outputs:
  v3_market_efficiency_birth.csv
  v3_market_efficiency_hs.csv

Run:  python scripts/v3_market_efficiency.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

CELL_DEG = 1.0
MIN_DRAFTEES = 25          # a market needs this many picks to be rate-stable
SHRINK_STRENGTH = 25       # empirical-Bayes pseudo-count toward national mean
HIGH_VOLUME = 40           # threshold for the "high volume" flag


def arg(flag, default, cast=int):
    if flag in sys.argv:
        try:
            return cast(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def load():
    src = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') \
        else 'v3_analysis.csv'
    if not os.path.exists(src):
        print("ERROR: no analysis CSV found.")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()
    df['reached_mlb'] = df['reached_mlb'].astype(int)

    max_round = arg('--max-round', None)
    if max_round is not None and 'draftRound' in df.columns:
        before = len(df)
        df = df[df['draftRound'] <= max_round].copy()
        print(f"Source: {src}   HS draftees, {COHORT_LABEL}, "
              f"rounds 1-{max_round}: {len(df):,} (from {before:,})\n")
    else:
        print(f"Source: {src}   HS draftees, {COHORT_LABEL}: {len(df):,}\n")
    return df


def analyze(df, lat_col, lon_col, city_col, state_col, label, out_csv,
            min_ct=MIN_DRAFTEES, hi_vol=HIGH_VOLUME):
    d = df[df[lat_col].notna() & df[lon_col].notna()].copy()
    d['cx'] = np.floor(d[lon_col] / CELL_DEG).astype(int)
    d['cy'] = np.floor(d[lat_col] / CELL_DEG).astype(int)

    national = d['reached_mlb'].mean()

    # most common city label per cell, for readability
    def top_city(g):
        s = (g[city_col].astype(str) + ', ' + g[state_col].astype(str))
        return s.value_counts().index[0] if len(s) else ''

    grp = d.groupby(['cx', 'cy'])
    agg = grp.agg(n=('reached_mlb', 'size'),
                  mlb=('reached_mlb', 'sum'),
                  lat=(lat_col, 'mean'),
                  lon=(lon_col, 'mean')).reset_index()
    agg['top_market'] = grp.apply(top_city, include_groups=False).values
    agg = agg[agg['n'] >= min_ct].copy()
    agg['raw_rate'] = agg['mlb'] / agg['n']
    agg['shrunk_rate'] = (agg['mlb'] + SHRINK_STRENGTH * national) / \
                         (agg['n'] + SHRINK_STRENGTH)
    # efficiency = shrunk rate relative to national; <1 under-converts
    agg['vs_national'] = agg['shrunk_rate'] / national

    agg = agg.sort_values('n', ascending=False)
    agg.to_csv(out_csv, index=False)

    print("=" * 88)
    print(f"MARKET EFFICIENCY BY {label.upper()}  "
          f"(national MLB rate = {national*100:.1f}%)")
    print("=" * 88)

    print(f"\nHIGHEST-VOLUME MARKETS (where the picks go):")
    print(f"{'Market':<26} {'Picks':>6} {'MLB':>5} {'Raw%':>6} {'Shrunk%':>8} "
          f"{'vs Nat':>7}")
    print("-" * 88)
    for _, r in agg.head(20).iterrows():
        flag = ''
        if r['n'] >= hi_vol and r['vs_national'] < 0.85:
            flag = '  <- high volume, LOW conversion'
        elif r['n'] >= hi_vol and r['vs_national'] > 1.15:
            flag = '  <- high volume, high conversion'
        print(f"{r['top_market'][:25]:<26} {int(r['n']):>6} {int(r['mlb']):>5} "
              f"{r['raw_rate']*100:>5.1f}% {r['shrunk_rate']*100:>7.1f}% "
              f"{r['vs_national']:>6.2f}x{flag}")

    print(f"\nLOWEST CONVERSION among high-volume markets (n>={hi_vol}) "
          f"-- possible OVER-coverage:")
    hv = agg[agg['n'] >= hi_vol].sort_values('shrunk_rate').head(12)
    print(f"{'Market':<26} {'Picks':>6} {'MLB':>5} {'Shrunk%':>8} {'vs Nat':>7}")
    print("-" * 88)
    for _, r in hv.iterrows():
        print(f"{r['top_market'][:25]:<26} {int(r['n']):>6} {int(r['mlb']):>5} "
              f"{r['shrunk_rate']*100:>7.1f}% {r['vs_national']:>6.2f}x")

    print(f"\nHIGHEST CONVERSION among high-volume markets (n>={hi_vol}) "
          f"-- efficient or possibly UNDER-covered:")
    eff = agg[agg['n'] >= hi_vol].sort_values('shrunk_rate', ascending=False).head(12)
    print(f"{'Market':<26} {'Picks':>6} {'MLB':>5} {'Shrunk%':>8} {'vs Nat':>7}")
    print("-" * 88)
    for _, r in eff.iterrows():
        print(f"{r['top_market'][:25]:<26} {int(r['n']):>6} {int(r['mlb']):>5} "
              f"{r['shrunk_rate']*100:>7.1f}% {r['vs_national']:>6.2f}x")

    print(f"\nSaved: {out_csv}\n")
    return agg


def main():
    df = load()
    # When restricted to early rounds, markets are smaller; allow lower floors.
    min_ct = arg('--min-count', MIN_DRAFTEES)
    hi_vol = arg('--high-volume', HIGH_VOLUME)

    have_birth = 'birth_lat' in df.columns and 'birth_lon' in df.columns
    have_hs = 'hs_lat' in df.columns and 'hs_lon' in df.columns

    if have_birth:
        analyze(df, 'birth_lat', 'birth_lon', 'birth_city', 'birth_state',
                'birth location', 'v3_market_efficiency_birth.csv',
                min_ct=min_ct, hi_vol=hi_vol)
    else:
        print("(no birth coordinates found; skipping birth analysis)")

    if have_hs:
        analyze(df, 'hs_lat', 'hs_lon', 'hs_city', 'hs_state',
                'high school location', 'v3_market_efficiency_hs.csv',
                min_ct=min_ct, hi_vol=hi_vol)
    else:
        print("(no HS coordinates found; skipping HS analysis)")

    print("Reminder: DESCRIPTIVE. Low conversion in a high-volume market is")
    print("CONSISTENT WITH over-coverage but does not prove it. Signability picks,")
    print("deep-round penetration, and draft-and-follow all also lower conversion.")
    print("Treat flagged markets as leads for a closer look, not conclusions.")
    print("Done.")


if __name__ == '__main__':
    main()
