"""
Fine-grained distance-bucket analysis (5-mile increments), 1996-2019.

Answers two distinct questions:
  1. SHAPE  - does the MLB reach rate keep climbing with distance, or does it
              jump off "same city" and then plateau?
  2. CUTPOINT - if a binary mover flag is wanted, where is the cleanest cut?

Design (per user choice):
  - exact-0 (same city) is its own bucket: the stayer baseline
  - then 0-5, 5-10, 10-15, ... in 5-mile steps ALL THE WAY to the max
    (the long tail is sparse and noisy; sample sizes are printed so the
    noise is visible rather than hidden)

For each bucket: n, MLB count, reach rate, Wilson 95% CI, and the odds ratio
vs the same-city baseline. A cumulative view then asks, for each candidate
threshold T, how movers (>T) compare with everyone at-or-below.

Outputs:
  v3_distance_buckets_5mi.csv
  v3_distance_threshold_scan.csv
  figures/fig22_distance_buckets_5mi.png

Input: v3_analysis_with_war.csv (or v3_analysis.csv)
Run:   python scripts/v3_distance_buckets.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

BUCKET_WIDTH = 5.0


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def odds_ratio(k1, n1, k0, n0):
    """OR of group1 vs group0, with 0.5 continuity correction if needed."""
    a, b = k1, n1 - k1
    c, d = k0, n0 - k0
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return orr, np.exp(np.log(orr) - 1.96*se), np.exp(np.log(orr) + 1.96*se)


def load():
    src = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') \
        else 'v3_analysis.csv'
    if not os.path.exists(src):
        print("ERROR: no analysis CSV found. Run the pipeline first.")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna()].copy()
    if 'year' in df.columns:
        df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    df['reached_mlb'] = df['reached_mlb'].astype(int)
    print(f"Source: {src}")
    print(f"HS draftees with distance, {COHORT_LABEL}: {len(df):,}\n")
    return df


def main():
    df = load()
    d = df['distance_miles'].values
    y = df['reached_mlb'].values
    max_d = d.max()

    # ---- baseline: same city (exactly 0) --------------------------------
    base_mask = d == 0
    kb, nb = int(y[base_mask].sum()), int(base_mask.sum())
    pb, pb_lo, pb_hi = wilson(kb, nb)
    print("=" * 92)
    print("DISTANCE BUCKETS (5-mile increments), MLB reach rate")
    print("=" * 92)
    print(f"{'Bucket (mi)':<16} {'N':>7} {'MLB':>6} {'Rate':>7} {'95% CI':>16} "
          f"{'OR vs same-city':>18}")
    print("-" * 92)
    print(f"{'same city (0)':<16} {nb:>7,} {kb:>6,} {pb*100:>6.1f}% "
          f"[{pb_lo*100:>4.1f}, {pb_hi*100:>4.1f}]  {'baseline':>18}")

    rows = [{'bucket': 'same_city', 'lo': 0.0, 'hi': 0.0, 'n': nb, 'mlb': kb,
             'rate': round(pb, 4), 'ci_lo': round(pb_lo, 4), 'ci_hi': round(pb_hi, 4),
             'or_vs_base': 1.0, 'or_lo': np.nan, 'or_hi': np.nan}]

    # ---- 5-mile buckets from >0 up to the max ---------------------------
    edges = np.arange(0, np.ceil(max_d / BUCKET_WIDTH) * BUCKET_WIDTH + BUCKET_WIDTH,
                      BUCKET_WIDTH)
    for lo, hi in zip(edges[:-1], edges[1:]):
        # (lo, hi]; the first bucket is (0,5] since exact-0 is separate
        m = (d > lo) & (d <= hi)
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        p, plo, phi = wilson(k, n)
        orr, olo, ohi = odds_ratio(k, n, kb, nb)
        flag = '' if n >= 30 else '  (small n)'
        print(f"{f'{lo:.0f}-{hi:.0f}':<16} {n:>7,} {k:>6,} {p*100:>6.1f}% "
              f"[{plo*100:>4.1f}, {phi*100:>4.1f}]  "
              f"{orr:>6.2f} [{olo:>4.2f}, {ohi:>4.2f}]{flag}")
        rows.append({'bucket': f'{lo:.0f}-{hi:.0f}', 'lo': lo, 'hi': hi, 'n': n,
                     'mlb': k, 'rate': round(p, 4), 'ci_lo': round(plo, 4),
                     'ci_hi': round(phi, 4), 'or_vs_base': round(orr, 4),
                     'or_lo': round(olo, 4), 'or_hi': round(ohi, 4)})

    buckets = pd.DataFrame(rows)
    buckets.to_csv('v3_distance_buckets_5mi.csv', index=False)

    # ---- threshold scan: mover = distance > T ---------------------------
    print(f"\n{'=' * 92}")
    print("THRESHOLD SCAN: if mover = (distance > T), how do movers vs the rest compare?")
    print("=" * 92)
    print(f"{'T (mi)':>7} {'%movers':>8} {'moverRate':>10} {'restRate':>9} "
          f"{'gap(pp)':>8} {'OR':>7} {'95% CI':>16}")
    print("-" * 92)
    scan = []
    for T in [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]:
        if T > max_d:
            break
        mv = d > T
        n1, n0 = int(mv.sum()), int((~mv).sum())
        if n1 == 0 or n0 == 0:
            continue
        k1, k0 = int(y[mv].sum()), int(y[~mv].sum())
        r1, r0 = k1/n1, k0/n0
        orr, olo, ohi = odds_ratio(k1, n1, k0, n0)
        print(f"{T:>7} {mv.mean()*100:>7.1f}% {r1*100:>9.1f}% {r0*100:>8.1f}% "
              f"{(r1-r0)*100:>+7.1f} {orr:>7.2f} [{olo:>4.2f}, {ohi:>4.2f}]")
        scan.append({'threshold_mi': T, 'pct_movers': round(mv.mean()*100, 2),
                     'mover_rate': round(r1, 4), 'rest_rate': round(r0, 4),
                     'gap_pp': round((r1-r0)*100, 2), 'odds_ratio': round(orr, 4),
                     'or_lo': round(olo, 4), 'or_hi': round(ohi, 4)})
    pd.DataFrame(scan).to_csv('v3_distance_threshold_scan.csv', index=False)

    print(f"\nSaved: v3_distance_buckets_5mi.csv, v3_distance_threshold_scan.csv")

    # ---- interpretation hint --------------------------------------------
    near = buckets[(buckets['lo'] > 0) & (buckets['n'] >= 30)]
    if len(near) >= 3:
        first_three = near.head(3)['rate'].mean()
        rest = near.iloc[3:]
        rest_mean = rest['rate'].mean() if len(rest) else np.nan
        print(f"\nShape check:")
        print(f"  same-city rate:            {pb*100:.1f}%")
        print(f"  mean rate, first 15 mi:    {first_three*100:.1f}%")
        if not np.isnan(rest_mean):
            print(f"  mean rate, beyond 15 mi:   {rest_mean*100:.1f}%")
        print("  If the jump is from same-city to the first bucket and it's flat")
        print("  after, that's a STEP function: any real move clears the bar and")
        print("  the exact cutoff barely matters.")

    # ---- figure ---------------------------------------------------------
    plot = buckets[buckets['n'] >= 20].copy()
    if len(plot) >= 3:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                                 'font.size': 11, 'axes.titlesize': 15,
                                 'axes.titleweight': 'bold',
                                 'axes.spines.top': False, 'axes.spines.right': False,
                                 'axes.grid': True, 'grid.alpha': 0.25})
            fig, ax = plt.subplots(figsize=(13, 6.5))
            xs = range(len(plot))
            rates = plot['rate'].values * 100
            lo_e = (plot['rate'] - plot['ci_lo']).values * 100
            hi_e = (plot['ci_hi'] - plot['rate']).values * 100
            colors = ['#C44E52' if b == 'same_city' else '#2C6FBB'
                      for b in plot['bucket']]
            ax.bar(xs, rates, yerr=[lo_e, hi_e], color=colors, capsize=2,
                   error_kw={'linewidth': 0.8, 'alpha': 0.6})
            ax.axhline(pb*100, ls='--', color='#C44E52', lw=1.2, alpha=0.8)
            ax.text(len(plot)-0.5, pb*100, ' same-city baseline', va='bottom',
                    ha='right', fontsize=9, color='#C44E52', style='italic')
            ax.set_xticks(list(xs))
            ax.set_xticklabels(plot['bucket'], rotation=60, ha='right', fontsize=8)
            ax.set_ylabel('Reached MLB (%)')
            ax.set_xlabel('Birth-to-high-school distance (miles), 5-mile buckets')
            ax.set_title('MLB reach rate by distance — where does "moving" start to matter?')
            fig.text(0.99, 0.01, f'HS draftees {COHORT_LABEL} (n={len(df):,}). '
                     f'Buckets with n<20 omitted from the chart. Bars = Wilson 95% CI. '
                     f'Red = same city.', ha='right', fontsize=8, color='#555')
            fig.tight_layout()
            os.makedirs('figures', exist_ok=True)
            p = 'figures/fig22_distance_buckets_5mi.png'
            fig.savefig(p, bbox_inches='tight'); plt.close(fig)
            print(f"  saved {p}")
        except ImportError:
            print("  (matplotlib unavailable; figure skipped)")
    print("\nDone.")


if __name__ == '__main__':
    main()
