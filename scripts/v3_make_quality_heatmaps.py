"""
Step 12 — Geographic QUALITY heatmaps (grid-cell level).

Not about the mover effect — this answers: which parts of the country
actually produce MLBers, and high-quality ones, per player drafted?

Aggregates players into lat/lon grid cells (metro-area-ish units), so each
cell pools enough players to be meaningful. Each cell's rate is shrunk toward
the national mean (empirical-Bayes) so lightly-sampled cells don't dominate.

Produces in figures/:
  fig13_heatmap_mlb_rate.png   MLB reach % per cell (shrunk)
  fig14_heatmap_war_quality.png  Career WAR per draftee per cell (shrunk)

Both use BIRTH location (where talent originates). Needs geopandas for borders
(falls back to borderless if unavailable). Needs WAR columns for fig14.

Input: v3_analysis_with_war.csv
Run:   python scripts/v3_make_quality_heatmaps.py
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
    from matplotlib.colors import Normalize
    from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

GEOJSON_LOCAL = 'us_states.geojson'  # reuse the cache from the geo-figures script
DROP_NONCONTINENTAL = {'AK', 'HI', 'PR'}

# Continental US window
LON_MIN, LON_MAX = -125, -66
LAT_MIN, LAT_MAX = 24, 50

# Grid resolution in degrees (~1.0 deg lon ~ 55 mi; metro-ish cells)
CELL_DEG = 1.0
MIN_PLAYERS_PER_CELL = 8     # cells with fewer players are not drawn
SHRINK_STRENGTH = 25         # pseudo-count for empirical-Bayes shrinkage
LABEL_TOP_N = 30            # label the N largest markets (by player count)


def darken(rgba, f=0.62):
    """Darken an RGBA color so it stays readable as text on a white label."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    return (r * f, g * f, b * f, 1.0)

# Major US metros (lat, lon) for labeling standout cells
METROS = {
    'Los Angeles': (34.05, -118.24), 'San Diego': (32.72, -117.16),
    'Bay Area': (37.77, -122.42), 'Sacramento': (38.58, -121.49),
    'Phoenix': (33.45, -112.07), 'Las Vegas': (36.17, -115.14),
    'Houston': (29.76, -95.37), 'Dallas': (32.78, -96.80),
    'San Antonio': (29.42, -98.49), 'Austin': (30.27, -97.74),
    'Miami': (25.76, -80.19), 'Tampa': (27.95, -82.46),
    'Orlando': (28.54, -81.38), 'Jacksonville': (30.33, -81.66),
    'Atlanta': (33.75, -84.39), 'Charlotte': (35.23, -80.84),
    'Nashville': (36.16, -86.78), 'New Orleans': (29.95, -90.07),
    'Chicago': (41.88, -87.63), 'Detroit': (42.33, -83.05),
    'Cleveland': (41.50, -81.69), 'Cincinnati': (39.10, -84.51),
    'St. Louis': (38.63, -90.20), 'Kansas City': (39.10, -94.58),
    'Minneapolis': (44.98, -93.27), 'Denver': (39.74, -104.99),
    'Seattle': (47.61, -122.33), 'Portland': (45.52, -122.68),
    'New York': (40.71, -74.01), 'Philadelphia': (39.95, -75.17),
    'Boston': (42.36, -71.06), 'Washington DC': (38.90, -77.04),
    'Pittsburgh': (40.44, -80.00), 'Indianapolis': (39.77, -86.16),
    'Columbus': (39.96, -82.99), 'Milwaukee': (43.04, -87.91),
    'Salt Lake City': (40.76, -111.89), 'Oklahoma City': (35.47, -97.52),
    'Memphis': (35.15, -90.05), 'Birmingham': (33.52, -86.81),
    'Raleigh': (35.78, -78.64), 'Norfolk': (36.85, -76.29),
}


def nearest_metro(lat, lon, max_deg=1.6):
    """Return (name, dist_deg) of the closest metro within max_deg, else (None, None)."""
    best, bestd = None, 1e9
    for name, (mlat, mlon) in METROS.items():
        d = np.hypot(lat - mlat, lon - mlon)
        if d < bestd:
            bestd, best = d, name
    if bestd <= max_deg:
        return best, bestd
    return None, None


def load_df():
    f = 'v3_analysis_with_war.csv'
    if not os.path.exists(f):
        print("ERROR: v3_analysis_with_war.csv not found. Run steps 1-6 first.")
        sys.exit(1)
    df = pd.read_csv(f, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['birth_lat'].notna() & df['birth_lon'].notna()].copy()
    df = df[(df['birth_lon'] > LON_MIN) & (df['birth_lon'] < LON_MAX) &
            (df['birth_lat'] > LAT_MIN) & (df['birth_lat'] < LAT_MAX)]
    return df


def get_states_gdf():
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if os.path.exists(GEOJSON_LOCAL):
        try:
            return gpd.read_file(GEOJSON_LOCAL)
        except Exception:
            return None
    # try a download (same sources as the geo-figures script)
    import urllib.request
    urls = [
        'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json',
        'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json',
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            with open(GEOJSON_LOCAL, 'w', encoding='utf-8') as f:
                f.write(data)
            return gpd.read_file(GEOJSON_LOCAL)
        except Exception:
            continue
    return None


def assign_cells(df):
    """Add integer cell indices for the lat/lon grid."""
    df = df.copy()
    df['cx'] = ((df['birth_lon'] - LON_MIN) / CELL_DEG).astype(int)
    df['cy'] = ((df['birth_lat'] - LAT_MIN) / CELL_DEG).astype(int)
    return df


def cell_aggregate(df, value_col, is_rate):
    """
    Aggregate to grid cells. Returns DataFrame with cell center lon/lat,
    n, raw value, and shrunk value.
    is_rate=True  -> value is mean of a 0/1 column (reach rate), shrunk to global mean
    is_rate=False -> value is mean of WAR-per-player, shrunk to global mean
    """
    g = df.groupby(['cx', 'cy'])
    agg = g.agg(n=(value_col, 'size'), s=(value_col, 'sum')).reset_index()
    agg = agg[agg['n'] >= MIN_PLAYERS_PER_CELL].copy()

    global_mean = df[value_col].mean()
    # Empirical-Bayes shrinkage toward the global mean
    agg['raw'] = agg['s'] / agg['n']
    agg['shrunk'] = (agg['s'] + SHRINK_STRENGTH * global_mean) / (agg['n'] + SHRINK_STRENGTH)

    agg['lon'] = LON_MIN + (agg['cx'] + 0.5) * CELL_DEG
    agg['lat'] = LAT_MIN + (agg['cy'] + 0.5) * CELL_DEG
    return agg, global_mean


def draw_heatmap(agg, gdf, value, title, cbar_label, fname, fmt_pct=False,
                 vmin=None, vmax=None, footnote=''):
    fig, ax = plt.subplots(figsize=(14, 8.5))
    if gdf is not None:
        g = gdf[~gdf.get('abbr', pd.Series([None]*len(gdf))).isin(DROP_NONCONTINENTAL)] \
            if 'abbr' in gdf.columns else gdf
        try:
            g.plot(color='#F7F7F7', edgecolor='#BBB', linewidth=0.6, ax=ax, zorder=1)
        except Exception:
            pass
    ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.axis('off')

    vals = agg[value].values
    if vmin is None:
        vmin = np.percentile(vals, 5)
    if vmax is None:
        vmax = np.percentile(vals, 95)

    # draw cells as squares colored by value
    cmap = plt.get_cmap('RdYlGn')
    norm = Normalize(vmin=vmin, vmax=vmax)
    half = CELL_DEG / 2
    for _, r in agg.iterrows():
        color = cmap(norm(r[value]))
        ax.add_patch(plt.Rectangle((r['lon'] - half, r['lat'] - half),
                                   CELL_DEG, CELL_DEG, facecolor=color,
                                   edgecolor='none', alpha=0.82, zorder=2))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
    cbar.set_label(cbar_label)
    if fmt_pct:
        cbar.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))

    # ---- label the largest markets by nearest metro, with light leader lines ----
    # Rank by cell size (number of players), dedup to distinct metros, take top N.
    cx, cy = (LON_MIN + LON_MAX) / 2, (LAT_MIN + LAT_MAX) / 2
    ranked = agg.sort_values('n', ascending=False)
    used_metros = set()
    n_labeled = 0
    for _, r in ranked.iterrows():
        if n_labeled >= LABEL_TOP_N:
            break
        name, _d = nearest_metro(r['lat'], r['lon'])
        if name is None or name in used_metros:
            continue
        used_metros.add(name)
        n_labeled += 1

        # value string + color matched to the polygon (darkened for legibility)
        val = r[value]
        val_str = f'{val*100:.0f}%' if fmt_pct else f'{val:.1f}'
        val_color = darken(cmap(norm(val)))

        # tiny sample-size line: numerator/denominator for rates, n=.. otherwise
        if fmt_pct:
            count_str = f"{int(r['s'])}/{int(r['n'])}"
        else:
            count_str = f"n={int(r['n'])}"

        # push label radially outward from map center toward the margin
        dx, dy = r['lon'] - cx, r['lat'] - cy
        dist = np.hypot(dx, dy) or 1.0
        ox = r['lon'] + (dx / dist) * 4.8
        oy = r['lat'] + (dy / dist) * 3.4
        ox = min(max(ox, LON_MIN + 1.5), LON_MAX - 1.5)
        oy = min(max(oy, LAT_MIN + 1.0), LAT_MAX - 1.0)

        name_area = TextArea(name + '  ',
                             textprops=dict(fontsize=7.5, fontweight='bold', color='#222'))
        val_area = TextArea(val_str,
                            textprops=dict(fontsize=7.5, fontweight='bold', color=val_color))
        top_row = HPacker(children=[name_area, val_area], align='center', pad=0, sep=0)
        count_area = TextArea(count_str,
                              textprops=dict(fontsize=5.5, color='#777'))
        packed = VPacker(children=[top_row, count_area], align='center', pad=0, sep=1)
        ab = AnnotationBbox(
            packed, (r['lon'], r['lat']), xybox=(ox, oy),
            xycoords='data', boxcoords='data', frameon=True, pad=0.3,
            arrowprops=dict(arrowstyle='-', color='#555', lw=0.6, alpha=0.75))
        ab.patch.set_facecolor('white')
        ab.patch.set_edgecolor('#999')
        ab.patch.set_linewidth(0.4)
        ab.patch.set_alpha(0.88)
        ab.set_zorder(6)
        ax.add_artist(ab)

    print(f"    ({n_labeled} metro labels drawn)")
    ax.set_title(title, fontsize=16, fontweight='bold')
    fig.text(0.99, 0.02, footnote, ha='right', fontsize=8, color='#555')
    fig.tight_layout()
    p = os.path.join(OUTDIR, fname)
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def main():
    print("Generating quality heatmaps...")
    df = load_df()
    print(f"  sample: {len(df):,} HS draftees with birth coords")
    df = assign_cells(df)

    gdf = get_states_gdf()
    if gdf is not None:
        # attach abbr so we can drop non-continental
        name_col = next((c for c in ['name','NAME','STATE_NAME','state']
                         if c in gdf.columns), None)
        if name_col:
            from_abbr = {
                'Alaska':'AK','Hawaii':'HI','Puerto Rico':'PR'}
            gdf['abbr'] = gdf[name_col].map(from_abbr).fillna('')
    else:
        print("  (no basemap; drawing cells without borders)")

    # ---- Figure 13: MLB reach rate ----
    agg_rate, gm_rate = cell_aggregate(df, 'reached_mlb', is_rate=True)
    print(f"  reach-rate cells (>= {MIN_PLAYERS_PER_CELL} players): {len(agg_rate)}; "
          f"national mean {gm_rate*100:.1f}%")
    top = agg_rate.sort_values('shrunk', ascending=False).head(5)
    draw_heatmap(
        agg_rate, gdf, 'shrunk',
        'Which areas are most likely to produce MLBers (per player drafted)',
        'MLB reach rate (shrunk)', 'fig13_heatmap_mlb_rate.png', fmt_pct=True,
        footnote=(f'Birth location, HS draftees {COHORT_LABEL}. Cells = {CELL_DEG} deg (~55 mi), '
                  f'min {MIN_PLAYERS_PER_CELL} players, rates shrunk toward national mean '
                  f'({gm_rate*100:.0f}%). Greener = higher reach rate.'))

    # ---- Figure 14: WAR per draftee ----
    if 'career_war' in df.columns:
        # WAR per drafted player: non-MLB / no-match treated as 0 WAR contribution
        df['war_contrib'] = df['career_war'].fillna(0)
        agg_war, gm_war = cell_aggregate(df, 'war_contrib', is_rate=False)
        print(f"  WAR cells: {len(agg_war)}; national mean {gm_war:.2f} WAR/draftee")
        draw_heatmap(
            agg_war, gdf, 'shrunk',
            'Which areas produce the highest-value talent (career WAR per draftee)',
            'Career WAR per draftee (shrunk)', 'fig14_heatmap_war_quality.png',
            fmt_pct=False,
            footnote=(f'Birth location, HS draftees {COHORT_LABEL}. Cells = {CELL_DEG} deg (~55 mi), '
                      f'min {MIN_PLAYERS_PER_CELL} players, WAR/draftee shrunk toward national '
                      f'mean ({gm_war:.2f}). Greener = higher average career value.'))
    else:
        print("  skip fig14 (no career_war column)")

    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
