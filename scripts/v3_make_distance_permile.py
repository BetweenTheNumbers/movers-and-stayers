"""
Step 13 — MLB% vs exact birth-to-HS distance, with NO distance buckets.

Distance is rounded to the nearest integer mile (1-mile resolution). For each
mile value that has at least one player, we compute the share who reached MLB.

Produces in figures/:
  fig15_mlb_pct_by_mile_bar.png        Bar chart, x = 0..max miles
  fig16_mlb_pct_by_mile_logpoints.png  Scatter on log-x; point size = # players

Because the far tail has very few players per mile, those points are pure
noise (0/50/100%). Point size on the log chart shows which values are
well-supported. A reference line marks the overall MLB rate.

Input: v3_analysis_with_war.csv (or v3_analysis.csv)
Run:   python scripts/v3_make_distance_permile.py
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
MOVER_C, ACCENT_C, NEUTRAL = '#2C6FBB', '#4C9F70', '#555555'
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


def permile_table(df):
    d = df.copy()
    d['mile'] = d['distance_miles'].round().astype(int)
    g = (d.groupby('mile')
         .agg(n=('reached_mlb', 'size'), mlb=('reached_mlb', 'sum'))
         .reset_index())
    g['pct'] = g['mlb'] / g['n'] * 100
    return g


def fig_bar(df, g, xmax, overall_pct):
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(g['mile'], g['pct'], width=1.0, color=MOVER_C, align='center')
    ax.axhline(overall_pct, color=ACCENT_C, lw=1.5, ls='--')
    ax.text(xmax, overall_pct + 1.5, f'overall {overall_pct:.0f}%',
            ha='right', va='bottom', color=ACCENT_C, fontsize=10, style='italic')
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Birth-to-high-school distance (miles)')
    ax.set_ylabel('Reached MLB')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title('MLB reach rate by exact distance (per mile, no buckets)')
    fig.text(0.99, 0.01,
             f'HS draftees {COHORT_LABEL} (n={len(df):,}). One bar per integer mile with ≥1 player. '
             f'Tall single bars in the tail are 1-2 player miles (noise), not signal.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig15_mlb_pct_by_mile_bar.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def fig_logpoints(df, g, xmax, overall_pct):
    gg = g[g['mile'] >= 1].copy()
    sizes = 5 + 3.0 * np.sqrt(gg['n'])
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sc = ax.scatter(gg['mile'], gg['pct'], s=sizes, alpha=0.45,
                    color=MOVER_C, edgecolor='none')
    ax.axhline(overall_pct, color=ACCENT_C, lw=1.5, ls='--')
    ax.text(xmax, overall_pct + 1.5, f'overall {overall_pct:.0f}%',
            ha='right', va='bottom', color=ACCENT_C, fontsize=10, style='italic')
    ax.set_xscale('log')
    ax.set_xlim(1, xmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Birth-to-high-school distance (miles, log scale)')
    ax.set_ylabel('Reached MLB')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title('MLB reach rate by exact distance (log scale; point size = # players)')

    # size legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MOVER_C,
                      alpha=0.5, markersize=np.sqrt(5 + 3.0*np.sqrt(c)),
                      label=f'{c} players at this mile')
               for c in [1, 10, 50]]
    ax.legend(handles=handles, frameon=False, loc='upper left',
              labelspacing=1.3, fontsize=9)
    fig.text(0.99, 0.01,
             f'HS draftees {COHORT_LABEL} (n={len(df):,}). Each point is one integer-mile value. '
             f'Big points (near 0) are reliable; tiny points in the tail are noise.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig16_mlb_pct_by_mile_logpoints.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def main():
    print("Generating per-mile distance charts...")
    df = load()
    g = permile_table(df)
    overall_pct = df['reached_mlb'].mean() * 100
    maxmile = int(df['distance_miles'].max())
    xmax = int(np.ceil((maxmile + 1) / 1000.0) * 1000)
    print(f"  sample: {len(df):,} players; max distance {maxmile:,} mi; x-axis 0..{xmax:,}")
    print(f"  distinct integer-mile values with players: {len(g):,}")
    fig_bar(df, g, xmax, overall_pct)
    fig_logpoints(df, g, xmax, overall_pct)
    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
