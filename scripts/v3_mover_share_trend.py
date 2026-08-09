"""
Mover share over time, with optional national migration overlay.

TWO QUESTIONS, in order:

  1. INTERNAL (the gate): does the share of draftees who are MOVERS change
     across the cohort window? If baseball families were equally mobile every
     year, there is nothing for a national trend to explain and any overlay is
     decorative. This part needs no external data.

  2. OVERLAY (only if part 1 shows movement): does the draftee mover share
     track national migration? Uses the USDA ERS county/metro population files:
        Population-Change-Metro-Status-2024.csv   (US / metro / nonmetro rates)
        Nonmetropolitan-Population-Change-2024.csv (natural change vs migration)

IMPORTANT — what this can and cannot be:
  These are ANNUAL NATIONAL rates. The mover flag is a CROSS-SECTIONAL
  INDIVIDUAL measure (did a family relocate over ~18 childhood years, anywhere).
  They cannot be joined at the player level -- there is no key, and a player's
  move happened years before his draft. This is a DESCRIPTIVE TREND OVERLAY
  only: two time series shown together. It is NOT a player-level feature and
  NOT a causal link. Correlation of two smooth national trends over 24
  autocorrelated points is weak evidence; treat it as suggestive at most.

Outputs:
  v3_mover_share_by_year.csv
  figures/fig23_mover_share_by_year.png            (internal)
  figures/fig24_mover_share_vs_migration.png       (overlay, if files present)

Run:
  python scripts/v3_mover_share_trend.py
  python scripts/v3_mover_share_trend.py --usda-dir .    # where the CSVs live
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

METRO_FILE = 'Population-Change-Metro-Status-2024.csv'
NONMETRO_FILE = 'Nonmetropolitan-Population-Change-2024.csv'


def arg(flag, default, cast=str):
    if flag in sys.argv:
        try:
            return cast(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    h = (z*np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / d
    return p, max(0, c-h), min(1, c+h)


def load_analysis():
    src = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') \
        else 'v3_analysis.csv'
    if not os.path.exists(src):
        print("ERROR: no analysis CSV found.")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna() & df['mover'].notna()].copy()
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    df['mover'] = df['mover'].astype(int)
    print(f"Source: {src}   HS draftees with distance, {COHORT_LABEL}: {len(df):,}\n")
    return df


def main():
    usda_dir = arg('--usda-dir', '.')
    df = load_analysis()

    # ---- 1. mover share by year -----------------------------------------
    rows = []
    for yr, g in df.groupby('year'):
        k, n = int(g['mover'].sum()), len(g)
        p, lo, hi = wilson(k, n)
        rows.append({'year': int(yr), 'n': n, 'movers': k,
                     'mover_share': round(p, 4), 'ci_lo': round(lo, 4),
                     'ci_hi': round(hi, 4),
                     'mean_distance': round(g['distance_miles'].mean(), 1),
                     'median_distance': round(g['distance_miles'].median(), 1)})
    ts = pd.DataFrame(rows).sort_values('year')
    ts.to_csv('v3_mover_share_by_year.csv', index=False)

    print("=" * 72)
    print("MOVER SHARE BY DRAFT YEAR")
    print("=" * 72)
    print(f"{'Year':>5} {'N':>6} {'Movers':>7} {'Share':>7} {'95% CI':>15} "
          f"{'MeanDist':>9}")
    print("-" * 72)
    for _, r in ts.iterrows():
        print(f"{int(r['year']):>5} {int(r['n']):>6,} {int(r['movers']):>7,} "
              f"{r['mover_share']*100:>6.1f}% "
              f"[{r['ci_lo']*100:>4.1f},{r['ci_hi']*100:>4.1f}] "
              f"{r['mean_distance']:>8.0f}mi")

    # trend test: is mover share changing?
    from scipy import stats
    yrs = ts['year'].values.astype(float)
    shares = ts['mover_share'].values
    slope, intercept, r, p, se = stats.linregress(yrs, shares)
    print(f"\nLinear trend in mover share: {slope*100:+.3f} pp/year, "
          f"r={r:.3f}, p={p:.4f}")
    first5 = ts.head(5)['mover_share'].mean()
    last5 = ts.tail(5)['mover_share'].mean()
    print(f"  First 5 yrs mean: {first5*100:.1f}%   "
          f"Last 5 yrs mean: {last5*100:.1f}%   "
          f"change: {(last5-first5)*100:+.1f} pp")
    if p < 0.05:
        direction = "RISING" if slope > 0 else "FALLING"
        print(f"  -> Mover share is {direction} over the window (p={p:.4f}). "
              f"An overlay is worth examining.")
    else:
        print(f"  -> No significant trend (p={p:.4f}). The draftee population's "
              f"mobility is roughly\n     flat, so a national-trend overlay would "
              f"be decorative, not explanatory.")

    # ---- 2. USDA overlay -------------------------------------------------
    mfile = os.path.join(usda_dir, METRO_FILE)
    nmfile = os.path.join(usda_dir, NONMETRO_FILE)
    usda = None
    if os.path.exists(mfile) or os.path.exists(nmfile):
        pieces = []
        if os.path.exists(mfile):
            m = pd.read_csv(mfile, encoding='utf-8-sig')
            m = m.rename(columns={'United States': 'us_popchg',
                                  'Nonmetropolitan': 'nonmetro_popchg',
                                  'Metropolitan': 'metro_popchg'})
            pieces.append(m.set_index('Year'))
        if os.path.exists(nmfile):
            nm = pd.read_csv(nmfile, encoding='utf-8-sig')
            nm = nm.rename(columns={'Total population change': 'nonmetro_total',
                                    'Natural change': 'nonmetro_natural',
                                    'Net migration': 'nonmetro_net_migration'})
            pieces.append(nm.set_index('Year')[['nonmetro_net_migration',
                                                'nonmetro_natural']])
        usda = pd.concat(pieces, axis=1).reset_index()
        usda = usda[(usda['Year'] >= START_YEAR) & (usda['Year'] <= END_YEAR)]
        print(f"\nUSDA migration data loaded: {len(usda)} years "
              f"({int(usda['Year'].min())}-{int(usda['Year'].max())})")

        # correlate mover share with each available national series
        merged = ts.merge(usda, left_on='year', right_on='Year', how='inner')
        print("\nCorrelation of mover share with national series")
        print("(Pearson on levels AND on first differences — the differenced")
        print(" version guards against two unrelated trends looking correlated):")
        for col in ['nonmetro_net_migration', 'us_popchg', 'metro_popchg',
                    'nonmetro_popchg']:
            if col not in merged.columns:
                continue
            sub = merged[['mover_share', col]].dropna()
            if len(sub) < 5:
                continue
            r_lvl = sub['mover_share'].corr(sub[col])
            d = sub.diff().dropna()
            r_dif = d['mover_share'].corr(d[col]) if len(d) > 3 else np.nan
            print(f"  {col:<24} levels r={r_lvl:+.3f}   diff r={r_dif:+.3f}")
        print("\n  Reminder: 24 autocorrelated points. The differenced r is the")
        print("  honest one. A high levels-r with a near-zero diff-r means the")
        print("  two series merely trend together, not co-move.")

    # ---- figures ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                             'font.size': 11, 'axes.titlesize': 14,
                             'axes.titleweight': 'bold',
                             'axes.spines.top': False, 'axes.spines.right': False,
                             'axes.grid': True, 'grid.alpha': 0.25})
        os.makedirs('figures', exist_ok=True)

        # fig23 — internal
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.errorbar(ts['year'], ts['mover_share']*100,
                    yerr=[(ts['mover_share']-ts['ci_lo'])*100,
                          (ts['ci_hi']-ts['mover_share'])*100],
                    fmt='o-', color='#2C6FBB', capsize=3, linewidth=1.5)
        z = np.polyfit(ts['year'], ts['mover_share']*100, 1)
        ax.plot(ts['year'], np.poly1d(z)(ts['year']), '--', color='#C44E52',
                lw=2, label=f'trend {z[0]:+.2f} pp/yr')
        ax.set_xlabel('Draft year'); ax.set_ylabel('Share of HS draftees who moved (%)')
        ax.set_title('Did the draftee population become more or less mobile?')
        ax.legend(frameon=False)
        fig.text(0.99, 0.01, f'HS draftees {COHORT_LABEL}. Bars = Wilson 95% CI.',
                 ha='right', fontsize=8, color='#555')
        fig.tight_layout()
        fig.savefig('figures/fig23_mover_share_by_year.png', bbox_inches='tight')
        plt.close(fig)
        print("\n  saved figures/fig23_mover_share_by_year.png")

        # fig24 — overlay
        if usda is not None and 'nonmetro_net_migration' in usda.columns:
            fig, ax1 = plt.subplots(figsize=(11, 6))
            ax1.plot(ts['year'], ts['mover_share']*100, 'o-', color='#2C6FBB',
                     lw=2, label='Draftee mover share (left)')
            ax1.set_xlabel('Year'); ax1.set_ylabel('Draftee mover share (%)',
                                                    color='#2C6FBB')
            ax1.tick_params(axis='y', labelcolor='#2C6FBB')
            ax2 = ax1.twinx()
            ax2.plot(usda['Year'], usda['nonmetro_net_migration'], 's--',
                     color='#4C9F70', lw=2, label='US nonmetro net migration (right)')
            ax2.axhline(0, color='#4C9F70', lw=0.6, alpha=0.4)
            ax2.set_ylabel('Nonmetro net migration (%/yr)', color='#4C9F70')
            ax2.tick_params(axis='y', labelcolor='#4C9F70')
            ax2.spines['top'].set_visible(False)
            ax1.set_title('Draftee mobility vs national nonmetro migration')
            fig.text(0.99, 0.005, 'Descriptive overlay only — annual national '
                     'rate vs cross-sectional individual measure. Not a causal link. '
                     'Two trends over 24 points.', ha='right', fontsize=7.5, color='#999')
            fig.tight_layout()
            fig.savefig('figures/fig24_mover_share_vs_migration.png',
                        bbox_inches='tight')
            plt.close(fig)
            print("  saved figures/fig24_mover_share_vs_migration.png")
    except ImportError:
        print("  (matplotlib unavailable; figures skipped)")

    print(f"\nSaved: v3_mover_share_by_year.csv")
    print("Done.")


if __name__ == '__main__':
    main()
