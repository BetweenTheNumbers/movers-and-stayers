"""
Step 9 — Generate presentation figures from the pipeline outputs.

Produces 5 PNGs in a figures/ subfolder:
  fig1_distance_doseresponse.png   MLB reach rate by birth-to-HS distance bin
  fig2_selection_thresholds.png    Mover vs stayer at each WAR threshold
  fig3_career_war_distribution.png Career WAR among MLB-reachers, movers vs stayers
  fig4_hs_quality_robustness.png   Mover odds ratio across HS-quality model specs
  fig5_signed_interaction.png      Mover effect split by signed status

Inputs:
  v3_analysis_with_war.csv         (preferred — has WAR columns)
  v3_analysis.csv                  (fallback for distance/signed figures)
  v3_cohort_hs_tier_results.csv    (for the robustness figure)

Run standalone:   python scripts/v3_make_figures.py
Or via pipeline:  python run_pipeline.py --only 9
"""

import os
import sys
import pandas as pd
from config import COHORT_LABEL
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')  # no display needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
except ImportError:
    print("ERROR: matplotlib is required. Install with:")
    print("    pip install matplotlib")
    sys.exit(1)

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.6,
})

MOVER_C  = '#2C6FBB'   # blue
STAYER_C = '#C44E52'   # red
ACCENT_C = '#4C9F70'   # green
NEUTRAL  = '#555555'

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

DIST_LABELS = {
    '00_exact': 'Same\ncity', '01_under5': '<5', '02_5_10': '5-10',
    '03_10_15': '10-15', '04_15_20': '15-20', '05_20_25': '20-25',
    '06_25_50': '25-50', '07_50_100': '50-100', '08_100_250': '100-250',
    '09_250_500': '250-500', '10_500_1000': '500-1k', '11_1000_plus': '1k+',
}
BIN_ORDER = list(DIST_LABELS.keys())


def load_data():
    if os.path.exists('v3_analysis_with_war.csv'):
        df = pd.read_csv('v3_analysis_with_war.csv', low_memory=False)
        has_war = True
    elif os.path.exists('v3_analysis.csv'):
        df = pd.read_csv('v3_analysis.csv', low_memory=False)
        has_war = False
    else:
        print("ERROR: no v3_analysis*.csv found. Run the pipeline first.")
        sys.exit(1)
    # Restrict to HS draftees with a computed distance — the analysis sample
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna() & df['mover'].notna()].copy()
    return df, has_war


def annotate_bars(ax, bars, fmt='{:.1f}%', offset=0.4):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + offset, fmt.format(h),
                ha='center', va='bottom', fontsize=10, fontweight='bold')


# ── Figure 1: distance dose-response ──────────────────────────────────────────
def fig1_distance(df):
    g = (df.groupby('distance_bin')['reached_mlb']
         .agg(['mean', 'count']).reindex(BIN_ORDER).dropna())
    rates = g['mean'] * 100
    labels = [DIST_LABELS[b] for b in g.index]
    colors = [STAYER_C if b == '00_exact' else MOVER_C for b in g.index]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(rates)), rates.values, color=colors, width=0.78)
    annotate_bars(ax, bars)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Reached MLB')
    ax.set_xlabel('Distance between birth city and high school city (miles)')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(rates.values) * 1.18)
    ax.set_title('Born farther from your high school, more likely to reach MLB')
    # baseline reference line at same-city rate
    base = rates.iloc[0]
    ax.axhline(base, ls='--', lw=1, color=NEUTRAL, alpha=0.7)
    ax.text(len(labels)-0.5, base + 0.3, f'Same-city baseline ({base:.0f}%)',
            ha='right', va='bottom', fontsize=9, color=NEUTRAL, style='italic')
    n = int(g['count'].sum())
    fig.text(0.99, 0.01, f'HS draftees {COHORT_LABEL} (n={n:,}). Red = stayers (same city).',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig1_distance_doseresponse.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 2: selection effect across WAR thresholds ──────────────────────────
def fig2_thresholds(df, has_war):
    if not has_war:
        print("  skip fig2 (needs WAR columns)")
        return
    outcomes = [('reached_mlb', 'Reached\nMLB'),
                ('had_1war_career', '1+ WAR'),
                ('had_3war_career', '3+ WAR'),
                ('had_10war_career', '10+ WAR'),
                ('had_25war_career', '25+ WAR')]
    mv = df[df['mover'] == 1]
    st = df[df['mover'] == 0]
    mv_rates, st_rates = [], []
    for col, _ in outcomes:
        m = mv[col].fillna(0).mean() * 100 if col != 'reached_mlb' else mv[col].mean()*100
        s = st[col].fillna(0).mean() * 100 if col != 'reached_mlb' else st[col].mean()*100
        mv_rates.append(m); st_rates.append(s)

    x = np.arange(len(outcomes))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w/2, mv_rates, w, label='Movers', color=MOVER_C)
    b2 = ax.bar(x + w/2, st_rates, w, label='Stayers', color=STAYER_C)
    annotate_bars(ax, b1, offset=0.15); annotate_bars(ax, b2, offset=0.15)
    ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in outcomes])
    ax.set_ylabel('% of HS draftees reaching outcome')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_title('The mover advantage is largest at the MLB threshold,\nand shrinks at higher career bars')
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, max(mv_rates) * 1.18)
    fig.text(0.99, 0.01,
             'Movers stay ahead at every level, but the gap narrows as the bar rises, '
             'consistent with a selection (not talent) effect.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig2_selection_thresholds.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 3: career WAR distribution among MLB-reachers ──────────────────────
def fig3_war_dist(df, has_war):
    if not has_war:
        print("  skip fig3 (needs WAR columns)")
        return
    mlb = df[(df['reached_mlb'] == 1) & (df['fg_any_match'] == 1) &
             (df['career_war'].notna())]
    mv = mlb[mlb['mover'] == 1]['career_war']
    st = mlb[mlb['mover'] == 0]['career_war']

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([st.values, mv.values], vert=True, widths=0.5,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor(STAYER_C); bp['boxes'][0].set_alpha(0.75)
    bp['boxes'][1].set_facecolor(MOVER_C);  bp['boxes'][1].set_alpha(0.75)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Stayers\n(n={len(st)})', f'Movers\n(n={len(mv)})'])
    ax.set_ylabel('Career WAR')
    ax.set_ylim(-3, 26)
    # mean markers + labels
    for i, vals in enumerate([st, mv], start=1):
        ax.scatter([i], [vals.mean()], color='black', zorder=5, marker='D', s=45)
        ax.text(i + 0.08, vals.mean(), f'mean {vals.mean():.2f}',
                va='center', fontsize=10)
    ax.set_title('Once they reach MLB, movers and stayers have identical careers')
    fig.text(0.99, 0.01,
             'Career WAR distributions overlap (p=0.84). Mobility predicts reaching MLB, '
             'not performance once there. Extreme outliers hidden for readability.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig3_career_war_distribution.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 4: HS-quality robustness ───────────────────────────────────────────
def fig4_robustness():
    f = 'v3_cohort_hs_tier_results.csv'
    if not os.path.exists(f) or os.path.getsize(f) < 10:
        print("  skip fig4 (v3_cohort_hs_tier_results.csv missing or empty — re-run step 8)")
        return
    try:
        res = pd.read_csv(f)
    except Exception as e:
        print(f"  skip fig4 (could not read {f}: {e})")
        return
    if len(res) == 0 or 'variable' not in res.columns:
        print(f"  skip fig4 ({f} has no usable rows)")
        return
    mv = res[res['variable'] == 'mover'].copy()
    if len(mv) == 0:
        print("  skip fig4 (no mover rows)")
        return
    # Short labels for models
    short = {'B1: BASELINE (no HS quality)': 'Baseline',
             'B2: + HS tier dummies (LEAVE-ONE-OUT)': '+ HS tier\n(LOO)',
             'B3: + log(HS LOO MLB count)': '+ log HS\n(LOO)',
             'B4: + HS tier dummies (PRIOR HISTORY)': '+ HS tier\n(prior)',
             'B5: + log(HS PRIOR MLB count)': '+ log HS\n(prior)',
             'B6: + both LOO and PRIOR (continuous)': '+ both'}
    mv['short'] = mv['model'].map(short).fillna(mv['model'])
    ors = mv['odds_ratio'].values
    labels = mv['short'].values

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = range(len(ors))
    ax.plot(xs, ors, '-o', color=ACCENT_C, markersize=10, linewidth=2)
    for x, o in zip(xs, ors):
        ax.text(x, o + 0.02, f'{o:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.axhline(1.0, ls='--', color=NEUTRAL, lw=1, alpha=0.7)
    ax.text(len(ors)-1, 1.005, 'OR = 1 (no effect)', ha='right', va='bottom',
            fontsize=9, color=NEUTRAL, style='italic')
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels)
    ax.set_ylabel('Mover odds ratio (reaching MLB)')
    ax.set_ylim(0.9, max(ors) * 1.15)
    ax.set_title('The mover effect survives controls for high-school quality')
    fig.text(0.99, 0.01,
             'Adding HS-program quality (two leakage-free measures) barely moves the '
             'mover odds ratio. The effect is not an elite-program artifact.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig4_hs_quality_robustness.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 5: signed vs unsigned interaction ──────────────────────────────────
def fig5_signed(df):
    if 'signed_flag' not in df.columns:
        print("  skip fig5 (no signed_flag)")
        return
    groups = [(1, 'Signed'), (0, 'Did not sign')]
    mv_rates, st_rates, ns = [], [], []
    for val, _ in groups:
        sub = df[df['signed_flag'] == val]
        mv_rates.append(sub[sub['mover'] == 1]['reached_mlb'].mean() * 100)
        st_rates.append(sub[sub['mover'] == 0]['reached_mlb'].mean() * 100)
        ns.append(len(sub))

    x = np.arange(len(groups))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, mv_rates, w, label='Movers', color=MOVER_C)
    b2 = ax.bar(x + w/2, st_rates, w, label='Stayers', color=STAYER_C)
    annotate_bars(ax, b1, offset=0.3); annotate_bars(ax, b2, offset=0.3)
    # gap brackets
    for i in range(len(groups)):
        gap = mv_rates[i] - st_rates[i]
        top = max(mv_rates[i], st_rates[i])
        ax.text(i, top + 2.5, f'+{gap:.1f} pp', ha='center', fontweight='bold',
                color=ACCENT_C, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{lbl}\n(n={n:,})' for (_, lbl), n in zip(groups, ns)])
    ax.set_ylabel('Reached MLB')
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(mv_rates) * 1.25)
    ax.set_title('The mover advantage is even larger among players who did not sign')
    ax.legend(frameon=False, loc='upper right')
    fig.text(0.99, 0.01,
             'Among non-signers (who often took the college route), the mover gap is wider, '
             'supporting a durable family-investment interpretation.',
             ha='right', fontsize=8, color=NEUTRAL)
    fig.tight_layout()
    path = os.path.join(OUTDIR, 'fig5_signed_interaction.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {path}")


def main():
    print("Generating figures...")
    df, has_war = load_data()
    print(f"  analysis sample: {len(df):,} HS draftees with distance")
    fig1_distance(df)
    fig2_thresholds(df, has_war)
    fig3_war_dist(df, has_war)
    fig4_robustness()
    fig5_signed(df)
    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
