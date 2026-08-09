"""
Step 15 — College-expansion figures.

Visualizes the HS-vs-college "washout" result from step 14:
the birth-to-HS mover effect is strong for HS draftees but largely washes
out for players drafted after college.

Produces in figures/:
  fig17_mover_by_path.png        Grouped mover/stayer reach rates: HS vs college
  fig18_mover_or_by_path.png     Mover odds ratio for HS vs college (with the interaction)

Input:
  v3_college_expansion_summary.csv     (from step 14)
  v3_college_interaction_logit.csv     (from step 14)
  v3_analysis_expanded.csv             (for confidence intervals / counts)

Run:  python scripts/v3_make_college_figures.py
"""

import os
import sys
import numpy as np
import pandas as pd

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

SUMMARY = 'v3_college_expansion_summary.csv'
LOGIT = 'v3_college_interaction_logit.csv'
EXPANDED = 'v3_analysis_expanded.csv'


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (p, center - half, center + half)


def annotate_bars(ax, bars, fmt='{:.1f}%', offset=0.4):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + offset, fmt.format(h),
                ha='center', va='bottom', fontsize=10, fontweight='bold')


# ── Figure 17: grouped mover/stayer reach rates by path ───────────────────────
def fig17(summary):
    order = ['HS draftees', 'College draftees', 'Combined (HS + college)']
    summary = summary.set_index('group').reindex([g for g in order if g in summary['group'].values]) \
        if 'group' in summary.columns else summary
    summary = pd.read_csv(SUMMARY).set_index('group')
    rows = [g for g in order if g in summary.index]

    mover = [summary.loc[g, 'mover_pct'] for g in rows]
    stayer = [summary.loc[g, 'stayer_pct'] for g in rows]
    diffs = [summary.loc[g, 'diff_pp'] for g in rows]

    x = np.arange(len(rows))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6.5))
    b1 = ax.bar(x - w/2, mover, w, label='Movers', color=MOVER_C)
    b2 = ax.bar(x + w/2, stayer, w, label='Stayers', color=STAYER_C)
    annotate_bars(ax, b1, offset=0.3)
    annotate_bars(ax, b2, offset=0.3)

    for i, g in enumerate(rows):
        top = max(mover[i], stayer[i])
        ax.text(i, top + 2.6, f'+{diffs[i]:.1f} pp', ha='center',
                fontweight='bold', color=ACCENT_C, fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels([g.replace(' (HS + college)', '') for g in rows])
    ax.set_ylabel('Reached MLB')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(mover) * 1.25)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('The mover advantage is large for HS draftees, small after college')
    fig.text(0.99, 0.01,
             'Birth-to-HS movers vs stayers. The gap nearly disappears for college draftees, '
             'consistent with the family-investment signal washing out over four college years.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig17_mover_by_path.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


# ── Figure 18: mover odds ratio by path (from the interaction model) ──────────
def fig18(logit):
    params = logit.set_index('variable')
    or_hs = np.exp(params.loc['mover', 'coef'])
    or_col = np.exp(params.loc['mover', 'coef'] + params.loc['mover_x_college', 'coef'])

    fig, ax = plt.subplots(figsize=(8.5, 6))
    labels = ['HS draftees', 'College draftees']
    ors = [or_hs, or_col]
    colors = [MOVER_C, '#9DB8D2']
    bars = ax.bar(labels, ors, color=colors, width=0.55)
    for b, o in zip(bars, ors):
        ax.text(b.get_x() + b.get_width()/2, o + 0.02, f'OR {o:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.axhline(1.0, ls='--', color=NEUTRAL, lw=1)
    ax.text(1.45, 1.01, 'OR = 1 (no effect)', ha='right', va='bottom',
            fontsize=9, color=NEUTRAL, style='italic')
    ax.set_ylabel('Mover odds ratio for reaching MLB')
    ax.set_ylim(0.9, max(ors) * 1.18)
    ip = params.loc['mover_x_college', 'p_value']
    ax.set_title('Birth-to-HS mobility predicts MLB much more for HS draftees')
    fig.text(0.99, 0.01,
             f'From one logistic model with a mover x college interaction '
             f'(interaction p={ip:.1e}). Controls: draft round, year. '
             f'OR>1 means movers reach MLB more often.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig18_mover_or_by_path.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def main():
    print("Generating college-expansion figures...")
    if not os.path.exists(SUMMARY) or not os.path.exists(LOGIT):
        print(f"ERROR: run step 14 first (need {SUMMARY} and {LOGIT}).")
        sys.exit(1)
    summary = pd.read_csv(SUMMARY)
    logit = pd.read_csv(LOGIT)
    fig17(summary)
    if 'mover_x_college' in set(logit['variable']):
        fig18(logit)
    else:
        print("  skip fig18 (no interaction term in logit file)")
    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
