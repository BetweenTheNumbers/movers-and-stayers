"""
Step 16 — Figure index.

Scans figures/ and produces:
  figures/FIGURE_INDEX.md       captioned list of every figure (for slide-building)
  figures/_contact_sheet.png    thumbnail montage of all figures

Run:  python scripts/v3_make_figure_index.py
"""

import os
import sys
import glob

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

OUTDIR = 'figures'

# One-line captions keyed by filename stem.
CAPTIONS = {
    'fig1_distance_doseresponse': 'MLB reach rate by birth-to-HS distance bin (the dose-response headline).',
    'fig2_selection_thresholds': 'Mover vs stayer at each WAR threshold; gap shrinks at higher bars (selection, not talent).',
    'fig3_career_war_distribution': 'Career WAR among MLB-reachers: movers and stayers identical (boxplot).',
    'fig4_hs_quality_robustness': 'Mover odds ratio stable across 6 HS-quality model specs.',
    'fig5_signed_interaction': 'Mover effect by signed status; larger among non-signers.',
    'fig6_distance_histogram': 'Raw birth-to-HS distance distribution (zoom + full-range log-y).',
    'fig7_logdistance_distribution': 'Log-distance distribution; stayer spike + mover hump.',
    'fig8_prob_mlb_vs_distance': 'P(reach MLB) vs distance with Wilson CIs + logistic fit.',
    'fig9_net_migration_choropleth': 'Net mover flow per state (green = net arrivals).',
    'fig10_migration_flows': 'Arrow flow map of largest cold-state to sunbelt moves.',
    'fig11_city_bubbles_hs': 'High-school-city bubbles: volume + MLB production by town.',
    'fig12_city_bubbles_birth': 'Birth-city bubbles: where drafted players originate.',
    'fig13_heatmap_mlb_rate': 'Grid-cell MLB reach rate (shrunk), top markets labeled with s/n.',
    'fig14_heatmap_war_quality': 'Grid-cell career WAR per draftee (shrunk), top markets labeled with n.',
    'fig15_mlb_pct_by_mile_bar': 'MLB% by exact mile (bar, no buckets).',
    'fig16_mlb_pct_by_mile_logpoints': 'MLB% by exact mile (log-x; point size = #players).',
    'fig17_mover_by_path': 'Mover vs stayer reach rate: HS vs college vs combined.',
    'fig18_mover_or_by_path': 'Mover odds ratio HS (1.66) vs college (1.14) from interaction model.',
    'fig19_threshold_sensitivity': 'Mover odds ratio across mover-cutoff choices (1-50 mi robustness).',
    'fig21_mover_or_by_year': 'Mover odds ratio by draft year with fitted trend (temporal test).',
    'fig20_year_window_sensitivity': 'Mover effect across start-year windows (1996/1998/2000/... to 2019).',
    'fig20_mover_by_year': 'Mover effect over draft years (travel-ball era trend).',
}


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    figs = sorted(glob.glob(os.path.join(OUTDIR, 'fig*.png')))
    if not figs:
        print(f"No figures found in {OUTDIR}/. Run the figure steps first.")
        sys.exit(1)
    print(f"Found {len(figs)} figures.")

    # Markdown index
    md = os.path.join(OUTDIR, 'FIGURE_INDEX.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write("# Figure Index — Draft Mobility V3\n\n")
        f.write(f"{len(figs)} figures. Captions are one-liners for slide-building.\n\n")
        for p in figs:
            s = stem(p)
            cap = CAPTIONS.get(s, '(no caption on file)')
            f.write(f"- **{os.path.basename(p)}** — {cap}\n")
    print(f"  saved {md}")

    # Contact sheet
    n = len(figs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.2))
    axes = axes.flatten() if n > 1 else [axes]
    for ax in axes:
        ax.axis('off')
    for i, p in enumerate(figs):
        try:
            img = mpimg.imread(p)
            axes[i].imshow(img)
            axes[i].set_title(os.path.basename(p), fontsize=8, fontweight='bold')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"{os.path.basename(p)}\n(load error)",
                         ha='center', va='center', fontsize=7)
    fig.suptitle('Draft Mobility V3 — all figures', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    sheet = os.path.join(OUTDIR, '_contact_sheet.png')
    fig.savefig(sheet, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved {sheet}")
    print(f"\nDone. Open {sheet} to see everything at a glance.")


if __name__ == '__main__':
    main()
