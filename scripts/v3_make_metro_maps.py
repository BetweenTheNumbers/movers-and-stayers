"""
Metro maps: draftees grouped into city markers, sized by volume and colored
by a chosen metric. Renders the geographic story behind the market-efficiency
and mover-rate findings.

Produces one PNG per (geography x metric) combination:
  geography: birth location, or high-school location
  metric:
    mlb_rate      - share reaching MLB (the conversion story)
    mover_rate    - share who are movers
    vs_national   - MLB rate relative to national (over/under-performing)
    volume        - draft count only (where the picks come from)

Marker SIZE = number of draftees (volume). Marker COLOR = the metric.
Only metros with >= MIN_DRAFTEES are drawn, so small noisy markets do not
clutter the map. Top markets are labeled.

Uses a simple lat/lon scatter over a US state outline if us_states.geojson is
present; otherwise a plain lat/lon plane (still readable).

Outputs (into figures/):
  fig25_metro_birth_mlb_rate.png
  fig26_metro_hs_mlb_rate.png
  fig27_metro_hs_mover_rate.png
  fig28_metro_hs_vs_national.png
  ... (named by geography + metric)

Run:  python scripts/v3_make_metro_maps.py
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

CELL_DEG = 1.0          # ~degree cell to group nearby draftees into a "metro"
MIN_DRAFTEES = 25       # metros below this are not drawn
LABEL_TOP = 18          # label this many highest-volume metros
SHRINK = 25             # empirical-Bayes strength for rate stabilization


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
    print(f"Source: {src}   HS draftees {COHORT_LABEL}: {len(df):,}")
    return df


def try_base_map(ax):
    """Draw US state outlines if geojson is available; else skip quietly."""
    for path in ('us_states.geojson', '../us_states.geojson'):
        if os.path.exists(path):
            try:
                import json
                from matplotlib.patches import Polygon
                from matplotlib.collections import PatchCollection
                gj = json.load(open(path))
                patches = []
                for feat in gj['features']:
                    geom = feat['geometry']
                    polys = (geom['coordinates'] if geom['type'] == 'MultiPolygon'
                             else [geom['coordinates']])
                    for poly in polys:
                        ring = np.array(poly[0])
                        if ring.ndim == 2 and ring.shape[1] >= 2:
                            patches.append(Polygon(ring[:, :2], closed=True))
                pc = PatchCollection(patches, facecolor='#f2f2f0',
                                     edgecolor='#cccccc', linewidths=0.5, zorder=0)
                ax.add_collection(pc)
                return True
            except Exception:
                return False
    return False


def build_metros(df, lat_col, lon_col, city_col, state_col):
    d = df[df[lat_col].notna() & df[lon_col].notna()].copy()
    d['cx'] = np.floor(d[lon_col] / CELL_DEG).astype(int)
    d['cy'] = np.floor(d[lat_col] / CELL_DEG).astype(int)
    national = d['reached_mlb'].mean()

    def top_city(g):
        s = (g[city_col].astype(str) + ', ' + g[state_col].astype(str))
        return s.value_counts().index[0] if len(s) else ''

    g = d.groupby(['cx', 'cy'])
    m = g.agg(n=('reached_mlb', 'size'),
              mlb=('reached_mlb', 'sum'),
              lat=(lat_col, 'mean'),
              lon=(lon_col, 'mean'),
              mover_rate=('mover', 'mean')).reset_index()
    m['top_market'] = g.apply(top_city, include_groups=False).values
    m = m[m['n'] >= MIN_DRAFTEES].copy()
    m['mlb_rate'] = m['mlb'] / m['n']
    m['shrunk_rate'] = (m['mlb'] + SHRINK*national) / (m['n'] + SHRINK)
    m['vs_national'] = m['shrunk_rate'] / national
    return m, national


METRICS = {
    'mlb_rate':   ('MLB reach rate', 'mlb_rate', 'RdYlGn', '%'),
    'mover_rate': ('Mover share', 'mover_rate', 'PuOr', '%'),
    'vs_national': ('MLB rate vs national', 'vs_national', 'RdYlGn', 'x'),
    'volume':     ('Draft volume', 'n', 'Blues', 'n'),
}


def draw(m, national, metric_key, geo_label, fname):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    title_metric, col, cmap, unit = METRICS[metric_key]
    fig, ax = plt.subplots(figsize=(14, 8.5))
    try_base_map(ax)

    # continental US window (drop AK/HI/PR outliers for readability)
    view = m[(m['lon'] > -130) & (m['lon'] < -65) &
             (m['lat'] > 23) & (m['lat'] < 50)]

    sizes = 18 + (view['n'] / view['n'].max()) * 900
    vals = view[col].values
    if metric_key == 'vs_national':
        vmin, vmax, center = 0.5, 1.5, 1.0
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        sc = ax.scatter(view['lon'], view['lat'], s=sizes, c=vals, cmap=cmap,
                        norm=norm, alpha=0.82, edgecolor='#333', linewidth=0.5,
                        zorder=3)
    else:
        sc = ax.scatter(view['lon'], view['lat'], s=sizes, c=vals, cmap=cmap,
                        alpha=0.82, edgecolor='#333', linewidth=0.5, zorder=3)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    if unit == '%':
        cbar.set_label(f'{title_metric} (share)')
    elif unit == 'x':
        cbar.set_label(f'{title_metric} (1.0 = national avg)')
    else:
        cbar.set_label(title_metric)

    # label the biggest markets
    for _, r in view.nlargest(LABEL_TOP, 'n').iterrows():
        ax.annotate(r['top_market'].split(',')[0], (r['lon'], r['lat']),
                    fontsize=7.5, ha='center', va='center', zorder=4,
                    xytext=(0, 9), textcoords='offset points',
                    color='#222',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none',
                              alpha=0.6))

    ax.set_title(f'{title_metric} by {geo_label} metro  ({COHORT_LABEL})',
                 fontsize=15, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_xlim(-127, -66); ax.set_ylim(24, 50)
    ax.set_aspect(1.3)
    note = (f'Marker size = draft volume. Metros with < {MIN_DRAFTEES} draftees '
            f'omitted. Continental US shown.')
    if metric_key == 'vs_national':
        note += f'  National MLB rate = {national*100:.1f}%.'
    fig.text(0.5, 0.02, note, ha='center', fontsize=8, color='#666')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    os.makedirs('figures', exist_ok=True)
    fig.savefig(f'figures/{fname}', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  saved figures/{fname}")


def main():
    df = load()
    if 'birth_lat' not in df.columns:
        print("ERROR: no coordinate columns; run the distance step first.")
        sys.exit(1)

    geos = [
        ('birth', 'birth_lat', 'birth_lon', 'birth_city', 'birth_state', 'birth'),
        ('hs', 'hs_lat', 'hs_lon', 'hs_city', 'hs_state', 'high-school'),
    ]
    fig_n = 25
    for gkey, la, lo, ci, st, glabel in geos:
        m, national = build_metros(df, la, lo, ci, st)
        print(f"\n{glabel}: {len(m)} metros with >= {MIN_DRAFTEES} draftees")
        for mkey in ['mlb_rate', 'mover_rate', 'vs_national', 'volume']:
            draw(m, national, mkey,
                 glabel, f'fig{fig_n}_metro_{gkey}_{mkey}.png')
            fig_n += 1

    print("\nDone.")


if __name__ == '__main__':
    main()
