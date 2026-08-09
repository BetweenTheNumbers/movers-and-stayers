"""
Step 10 — Distance distribution + probability curves (matplotlib only).

Produces 3 PNGs in figures/:
  fig6_distance_histogram.png      Raw birth-to-HS distance distribution
  fig7_logdistance_distribution.png  Log-distance distribution (spike at 0 + mover hump)
  fig8_prob_mlb_vs_distance.png    Smoothed P(reach MLB) vs distance, with CI band

Input: v3_analysis_with_war.csv (or v3_analysis.csv)
Run:   python scripts/v3_make_distance_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd
from config import COHORT_LABEL

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 150, 'font.size': 12,
    'axes.titlesize': 15, 'axes.titleweight': 'bold', 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
})
MOVER_C, STAYER_C, ACCENT_C, NEUTRAL = '#2C6FBB', '#C44E52', '#4C9F70', '#555555'
OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)


def load():
    f = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') else 'v3_analysis.csv'
    if not os.path.exists(f):
        print("ERROR: no analysis CSV found. Run the pipeline first.")
        sys.exit(1)
    df = pd.read_csv(f, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna()].copy()
    return df


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (p, center - half, center + half)


# ── Figure 6: raw distance histogram (two panels) ─────────────────────────────
def fig6(df):
    d = df['distance_miles'].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: zoomed 0-100 miles
    sub = d[d <= 100]
    ax1.hist(sub, bins=40, color=MOVER_C, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax1.axvline(5, color=STAYER_C, ls='--', lw=1.5)
    ax1.text(6, ax1.get_ylim()[1]*0.92, '5-mi mover\nthreshold',
             color=STAYER_C, fontsize=9, va='top')
    ax1.set_xlabel('Distance (miles)')
    ax1.set_ylabel('Number of players')
    ax1.set_title('Close range (0-100 miles)', fontsize=13)

    # Right: full range, log y
    ax2.hist(d, bins=60, color=NEUTRAL, alpha=0.85, edgecolor='white', linewidth=0.4)
    ax2.set_yscale('log')
    ax2.set_xlabel('Distance (miles)')
    ax2.set_ylabel('Number of players (log scale)')
    ax2.set_title('Full range (log scale)', fontsize=13)

    fig.suptitle('Most players are born near their high school, with a long mobility tail',
                 fontsize=15, fontweight='bold')
    med = np.median(d)
    fig.text(0.99, 0.01,
             f'HS draftees {COHORT_LABEL} (n={len(d):,}). Median distance {med:.0f} mi; '
             f'{(d<=5).mean()*100:.0f}% within 5 miles (stayers).',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    p = os.path.join(OUTDIR, 'fig6_distance_histogram.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


# ── Figure 7: log-distance distribution ───────────────────────────────────────
def fig7(df):
    d = df['distance_miles'].values
    logd = np.log10(d + 1)  # log10(miles + 1)
    fig, ax = plt.subplots(figsize=(11, 6))
    n, bins, patches = ax.hist(logd, bins=50, color=MOVER_C, alpha=0.85,
                               edgecolor='white', linewidth=0.4)
    # Color the same-city spike (log10(1)=0 region) red
    for patch, left in zip(patches, bins[:-1]):
        if left < np.log10(5 + 1):
            patch.set_facecolor(STAYER_C)

    # x ticks at human distances
    ticks_mi = [0, 1, 5, 10, 50, 100, 500, 1000, 3000]
    ax.set_xticks([np.log10(t + 1) for t in ticks_mi])
    ax.set_xticklabels([str(t) for t in ticks_mi])
    ax.set_xlabel('Distance (miles, log scale)')
    ax.set_ylabel('Number of players')
    ax.axvline(np.log10(5 + 1), color='black', ls='--', lw=1.2)
    ax.set_title('Two populations: a "stayer" spike near zero and a broad "mover" hump')
    # legend proxies
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STAYER_C, label='Stayers (≤5 mi)'),
                       Patch(color=MOVER_C, label='Movers (>5 mi)')],
              frameon=False, loc='upper right')
    fig.text(0.99, 0.01,
             f'HS draftees {COHORT_LABEL} (n={len(d):,}). Log scale reveals the bimodal structure '
             f'hidden in the raw histogram.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig7_logdistance_distribution.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


# ── Figure 8: P(MLB) vs distance, empirical + logistic fit ────────────────────
def fig8(df):
    d = df['distance_miles'].values
    y = df['reached_mlb'].values

    # Empirical binned rates with Wilson CI (quantile-based bins for balance)
    n_bins = 12
    # Use quantile edges but ensure the same-city group (dist<=1) is its own bin
    qs = np.quantile(d[d > 1], np.linspace(0, 1, n_bins))
    edges = np.unique(np.concatenate([[0, 1.0001], qs]))
    centers, rates, los, his, counts = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (d >= lo) & (d < hi)
        nn = mask.sum()
        if nn < 20:
            continue
        kk = y[mask].sum()
        p, l, h = wilson(kk, nn)
        # geometric-ish center for log axis
        c = max(np.median(d[mask]), 0.5)
        centers.append(c); rates.append(p*100); los.append(l*100); his.append(h*100)
        counts.append(nn)

    centers = np.array(centers); rates = np.array(rates)
    los = np.array(los); his = np.array(his)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    # empirical points with error bars
    ax.errorbar(centers, rates, yerr=[rates-los, his-rates], fmt='o',
                color=MOVER_C, markersize=7, capsize=3, linewidth=1.2,
                label='Empirical reach rate (95% CI)', zorder=4)

    # logistic fit on log_distance
    try:
        import statsmodels.api as sm
        logd = np.log1p(d)
        X = sm.add_constant(logd)
        m = sm.Logit(y, X).fit(disp=False)
        grid = np.linspace(max(d.min(), 0), np.quantile(d, 0.995), 300)
        Xg = sm.add_constant(np.log1p(grid))
        pred = m.predict(Xg) * 100
        ax.plot(np.clip(grid, 0.5, None), pred, color=ACCENT_C, lw=2.5,
                label='Logistic fit (P ~ log distance)', zorder=3)
    except Exception as e:
        print(f"  (logistic fit skipped: {e})")

    ax.set_xscale('log')
    ax.set_xticks([1, 5, 10, 50, 100, 500, 1000])
    ax.set_xticklabels(['0-1', '5', '10', '50', '100', '500', '1000'])
    ax.set_xlabel('Distance between birth city and high school city (miles, log scale)')
    ax.set_ylabel('Probability of reaching MLB')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(his)*1.1)
    ax.set_title('Probability of reaching MLB rises with distance, then plateaus')
    ax.legend(frameon=False, loc='lower right')
    fig.text(0.99, 0.01,
             f'HS draftees {COHORT_LABEL} (n={len(d):,}). Sharpest gain is the jump off zero; '
             f'beyond ~25 miles additional distance adds little.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig8_prob_mlb_vs_distance.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def main():
    print("Generating distance figures...")
    df = load()
    print(f"  sample: {len(df):,} HS draftees with distance")
    fig6(df); fig7(df); fig8(df)
    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
