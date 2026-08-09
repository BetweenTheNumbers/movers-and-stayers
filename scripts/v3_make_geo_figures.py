"""
Step 11 — Geographic net-migration maps (needs geopandas).

Shows where future MLB draftees were born vs where they went to high school,
focused on the cold-state -> sunbelt migration story.

Produces in figures/:
  fig9_net_migration_choropleth.png  Net mover flow per state (inflow - outflow)
  fig10_migration_flows.png          Arrow flow map of the largest cold->warm moves

Requires:
  pip install geopandas matplotlib

On first run it downloads a US-states GeoJSON and caches it locally as
us_states.geojson so later runs are offline. If the download fails, the
script falls back to a centroid-only plot (no state borders) using a
built-in coordinate table, so it still produces output.

Input: v3_analysis_with_war.csv (or v3_analysis.csv)
Run:   python scripts/v3_make_geo_figures.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from config import COHORT_LABEL

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib required. pip install matplotlib")
    sys.exit(1)

OUTDIR = 'figures'
os.makedirs(OUTDIR, exist_ok=True)

GEOJSON_LOCAL = 'us_states.geojson'
GEOJSON_URLS = [
    'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json',
    'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json',
]

WARM_STATES = {'FL','TX','CA','AZ','GA','NC','SC','AL','MS','LA','NV','HI','PR'}

# Continental-US state centroids (lat, lon) — fallback + arrow endpoints
STATE_CENTROIDS = {
    'AL': (32.8, -86.8), 'AZ': (34.2, -111.7), 'AR': (34.9, -92.4), 'CA': (37.2, -119.4),
    'CO': (39.0, -105.5), 'CT': (41.6, -72.7), 'DE': (39.0, -75.5), 'FL': (28.6, -81.5),
    'GA': (32.6, -83.4), 'IA': (42.0, -93.5), 'ID': (44.4, -114.6), 'IL': (40.0, -89.2),
    'IN': (39.9, -86.3), 'KS': (38.5, -98.4), 'KY': (37.5, -85.3), 'LA': (31.0, -92.0),
    'MA': (42.3, -71.8), 'MD': (39.0, -76.8), 'ME': (45.4, -69.2), 'MI': (44.3, -85.4),
    'MN': (46.3, -94.3), 'MO': (38.4, -92.5), 'MS': (32.7, -89.7), 'MT': (47.0, -109.6),
    'NC': (35.5, -79.4), 'ND': (47.5, -100.5), 'NE': (41.5, -99.8), 'NH': (43.7, -71.6),
    'NJ': (40.1, -74.7), 'NM': (34.4, -106.1), 'NV': (39.3, -116.6), 'NY': (42.9, -75.5),
    'OH': (40.3, -82.8), 'OK': (35.6, -97.5), 'OR': (44.0, -120.6), 'PA': (40.9, -77.8),
    'RI': (41.7, -71.6), 'SC': (33.9, -80.9), 'SD': (44.4, -100.2), 'TN': (35.9, -86.4),
    'TX': (31.5, -99.3), 'UT': (39.3, -111.7), 'VA': (37.5, -78.9), 'VT': (44.1, -72.7),
    'WA': (47.4, -120.5), 'WI': (44.6, -89.9), 'WV': (38.6, -80.6), 'WY': (43.0, -107.6),
    'DC': (38.9, -77.0),
}

STATE_NAME_TO_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','District of Columbia':'DC',
    'Florida':'FL','Georgia':'GA','Hawaii':'HI','Idaho':'ID','Illinois':'IL',
    'Indiana':'IN','Iowa':'IA','Kansas':'KS','Kentucky':'KY','Louisiana':'LA',
    'Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN',
    'Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV',
    'New Hampshire':'NH','New Jersey':'NJ','New Mexico':'NM','New York':'NY',
    'North Carolina':'NC','North Dakota':'ND','Ohio':'OH','Oklahoma':'OK','Oregon':'OR',
    'Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC','South Dakota':'SD',
    'Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA',
    'Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY','Puerto Rico':'PR',
}

DROP_NONCONTINENTAL = {'AK', 'HI', 'PR'}  # for clean continental map


def load_df():
    f = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') else 'v3_analysis.csv'
    if not os.path.exists(f):
        print("ERROR: no analysis CSV found. Run the pipeline first.")
        sys.exit(1)
    df = pd.read_csv(f, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    # Need both states, US-domestic
    df = df[df['birth_state'].notna() & df['hs_state'].notna()].copy()
    df['birth_state'] = df['birth_state'].str.upper().str.strip()
    df['hs_state'] = df['hs_state'].str.upper().str.strip()
    valid = set(STATE_CENTROIDS) | {'AK', 'HI', 'PR'}
    df = df[df['birth_state'].isin(valid) & df['hs_state'].isin(valid)]
    return df


def get_states_gdf():
    """Return a geopandas GeoDataFrame of US states, or None if unavailable."""
    try:
        import geopandas as gpd
    except ImportError:
        print("  geopandas not installed -> using centroid fallback (no borders).")
        print("  For borders: pip install geopandas")
        return None

    # Cached local file?
    if os.path.exists(GEOJSON_LOCAL):
        try:
            return gpd.read_file(GEOJSON_LOCAL)
        except Exception:
            pass

    # Try to download
    import urllib.request
    for url in GEOJSON_URLS:
        try:
            print(f"  downloading basemap: {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            with open(GEOJSON_LOCAL, 'w', encoding='utf-8') as f:
                f.write(data)
            return gpd.read_file(GEOJSON_LOCAL)
        except Exception as e:
            print(f"    failed: {e}")
    print("  could not fetch basemap -> centroid fallback (no borders).")
    return None


def attach_abbr(gdf):
    """Add a 'abbr' column to the states GeoDataFrame."""
    name_col = None
    for c in ['name', 'NAME', 'STATE_NAME', 'state']:
        if c in gdf.columns:
            name_col = c
            break
    if name_col is None:
        # try the first string column
        for c in gdf.columns:
            if gdf[c].dtype == object and c != 'geometry':
                name_col = c
                break
    gdf['abbr'] = gdf[name_col].map(STATE_NAME_TO_ABBR)
    return gdf


def compute_migration(df):
    """Per-state inflow/outflow/net of movers; plus flow pairs."""
    movers = df[df['birth_state'] != df['hs_state']]
    outflow = movers.groupby('birth_state').size()
    inflow = movers.groupby('hs_state').size()
    states = sorted(set(outflow.index) | set(inflow.index))
    rows = []
    for s in states:
        i = int(inflow.get(s, 0)); o = int(outflow.get(s, 0))
        rows.append({'state': s, 'inflow': i, 'outflow': o, 'net': i - o,
                     'warm': s in WARM_STATES})
    net_df = pd.DataFrame(rows)
    # Flow pairs
    flows = (movers.groupby(['birth_state', 'hs_state']).size()
             .reset_index(name='n').sort_values('n', ascending=False))
    return net_df, flows


# ── Figure 9: net migration choropleth ────────────────────────────────────────
def fig9(net_df, gdf):
    fig, ax = plt.subplots(figsize=(13, 8))
    if gdf is not None:
        g = gdf[~gdf['abbr'].isin(DROP_NONCONTINENTAL)].copy()
        g = g.merge(net_df, left_on='abbr', right_on='state', how='left')
        g['net'] = g['net'].fillna(0)
        vmax = max(abs(g['net'].min()), abs(g['net'].max()), 1)
        # RdYlGn: red = net loss, green = net gain (more players arriving)
        g.plot(column='net', cmap='RdYlGn', vmin=-vmax, vmax=vmax,
               linewidth=0.6, edgecolor='#888', ax=ax, legend=True,
               legend_kwds={'label': 'Net mover flow (green = more arriving)',
                            'shrink': 0.5})
        ax.set_xlim(-126, -66); ax.set_ylim(24, 50)
        ax.axis('off')
    else:
        # centroid fallback: bubble size = |net|, color = sign
        for _, r in net_df.iterrows():
            if r['state'] in DROP_NONCONTINENTAL or r['state'] not in STATE_CENTROIDS:
                continue
            lat, lon = STATE_CENTROIDS[r['state']]
            color = '#4C9F70' if r['net'] >= 0 else '#C44E52'  # green gain / red loss
            ax.scatter(lon, lat, s=20 + abs(r['net'])*4, color=color, alpha=0.75,
                       edgecolor='white', linewidth=0.5)
            ax.text(lon, lat, r['state'], ha='center', va='center', fontsize=7)
        ax.set_xlim(-126, -66); ax.set_ylim(24, 50)
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_title('Sunbelt states gain future MLB draftees; cold states lose them',
                 fontsize=16, fontweight='bold')
    fig.text(0.99, 0.02,
             'Net = (players who moved INTO the state for HS) − (born there but moved away). '
             'Green = net gain, red = net loss. AK/HI/PR excluded.',
             ha='right', fontsize=8, color='#555')
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig9_net_migration_choropleth.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


# ── Figure 10: migration flow arrows ──────────────────────────────────────────
def fig10(flows, gdf, top_n=30):
    fig, ax = plt.subplots(figsize=(13, 8))
    # Basemap
    if gdf is not None:
        g = gdf[~gdf['abbr'].isin(DROP_NONCONTINENTAL)].copy()
        g.plot(color='#EFEFEF', edgecolor='#BBB', linewidth=0.6, ax=ax)
    ax.set_xlim(-126, -66); ax.set_ylim(24, 50)
    ax.axis('off')

    # Focus on cold -> warm flows (the story), then biggest others
    flows = flows.copy()
    flows['cold_to_warm'] = (~flows['birth_state'].isin(WARM_STATES)) & \
                            (flows['hs_state'].isin(WARM_STATES))
    cold_warm = flows[flows['cold_to_warm']].head(top_n)

    maxn = cold_warm['n'].max() if len(cold_warm) else 1
    for _, r in cold_warm.iterrows():
        bs, hs = r['birth_state'], r['hs_state']
        if bs not in STATE_CENTROIDS or hs not in STATE_CENTROIDS:
            continue
        blat, blon = STATE_CENTROIDS[bs]
        hlat, hlon = STATE_CENTROIDS[hs]
        lw = 0.5 + 4.0 * (r['n'] / maxn)
        ax.annotate('', xy=(hlon, hlat), xytext=(blon, blat),
                    arrowprops=dict(arrowstyle='-|>', color='#C44E52',
                                    alpha=0.6, linewidth=lw,
                                    connectionstyle='arc3,rad=0.15'))

    # Mark destination (warm) states
    for s in WARM_STATES:
        if s in STATE_CENTROIDS and s not in DROP_NONCONTINENTAL:
            lat, lon = STATE_CENTROIDS[s]
            ax.scatter(lon, lat, s=60, color='#4C9F70', zorder=5,
                       edgecolor='white', linewidth=1)
            ax.text(lon, lat - 0.8, s, ha='center', va='top', fontsize=8,
                    fontweight='bold', color='#2A6B45')

    ax.set_title('Where families relocate for baseball: cold-state births → sunbelt high schools',
                 fontsize=15, fontweight='bold')
    legend_elems = [
        Line2D([0], [0], color='#C44E52', lw=3, label='Migration flow (width = # players)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C9F70',
               markersize=10, label='Sunbelt destination state'),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc='lower left', fontsize=10)
    fig.text(0.99, 0.02,
             f'Top {top_n} cold-state-to-sunbelt flows among HS draftees {COHORT_LABEL}. '
             f'Arrows point from birth state to high-school state.',
             ha='right', fontsize=8, color='#555')
    fig.tight_layout()
    p = os.path.join(OUTDIR, 'fig10_migration_flows.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def fig_city_bubbles(df, gdf, which, fignum):
    """
    City/town-level bubble map on state borders.
    which = 'hs'  -> bubbles at high-school cities (where players trained)
    which = 'birth' -> bubbles at birth cities (where players came from)
    Bubble size = number of HS draftees from that city.
    Bubble color (Greens) = number who reached MLB.
    """
    lat_col = f'{which}_lat'
    lon_col = f'{which}_lon'
    city_col = 'hs_city' if which == 'hs' else 'birth_city'
    state_col = 'hs_state' if which == 'hs' else 'birth_state'

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"  skip fig{fignum} ({lat_col}/{lon_col} not found — run v3_compute_distances first)")
        return

    sub = df[df[lat_col].notna() & df[lon_col].notna()].copy()
    # aggregate to the city level
    grp = (sub.groupby([city_col, state_col])
           .agg(lat=(lat_col, 'median'), lon=(lon_col, 'median'),
                n=('reached_mlb', 'size'), mlb=('reached_mlb', 'sum'))
           .reset_index())
    # continental US window for a clean map
    grp = grp[(grp['lon'] > -126) & (grp['lon'] < -66) &
              (grp['lat'] > 24) & (grp['lat'] < 50)]
    grp = grp.sort_values('n', ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8.5))
    if gdf is not None:
        g = gdf[~gdf['abbr'].isin(DROP_NONCONTINENTAL)].copy()
        g.plot(color='#F5F5F5', edgecolor='#BBB', linewidth=0.6, ax=ax, zorder=1)
    ax.set_xlim(-126, -66); ax.set_ylim(24, 50)
    ax.axis('off')

    # size scaling: area proportional to count
    sizes = 12 + grp['n'].values * 6
    sc = ax.scatter(grp['lon'], grp['lat'], s=sizes, c=grp['mlb'],
                    cmap='Greens', alpha=0.72, edgecolor='#2A6B45',
                    linewidth=0.4, zorder=3,
                    vmin=0, vmax=max(grp['mlb'].quantile(0.98), 1))
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label('Players who reached MLB (per city)')

    # label the top cities by count
    for _, r in grp.head(12).iterrows():
        ax.text(r['lon'], r['lat'] + 0.35, str(r[city_col]),
                ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#222', zorder=4)

    # size legend (proxy bubbles)
    from matplotlib.lines import Line2D
    legend_counts = [5, 20, 50]
    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor='#4C9F70', markeredgecolor='#2A6B45',
                      markersize=np.sqrt(12 + cnt*6), label=f'{cnt} draftees')
               for cnt in legend_counts]
    ax.legend(handles=handles, frameon=False, loc='lower left',
              labelspacing=1.4, title='Bubble size', fontsize=9)

    where_txt = 'high-school cities (where they trained)' if which == 'hs' \
                else 'birth cities (where they came from)'
    ax.set_title(f'MLB-draftee {where_txt}, by town',
                 fontsize=16, fontweight='bold')
    fig.text(0.99, 0.02,
             f'HS draftees {COHORT_LABEL} (n={int(grp["n"].sum()):,} across {len(grp):,} towns). '
             f'Bubble size = players from that town; greener = more reached MLB. '
             f'Continental US only.',
             ha='right', fontsize=8, color='#555')
    fig.tight_layout()
    p = os.path.join(OUTDIR, f'fig{fignum}_city_bubbles_{which}.png')
    fig.savefig(p, bbox_inches='tight'); plt.close(fig)
    print(f"  saved {p}")


def main():
    print("Generating geographic figures...")
    df = load_df()
    print(f"  sample: {len(df):,} HS draftees with both states")
    net_df, flows = compute_migration(df)

    # Quick text summary of the migration story
    warm_net = net_df[net_df['warm']]['net'].sum()
    cold_net = net_df[~net_df['warm']]['net'].sum()
    print(f"  Net mover flow INTO sunbelt states:  {warm_net:+d}")
    print(f"  Net mover flow into non-sunbelt:     {cold_net:+d}")
    top = net_df.sort_values('net', ascending=False)
    print("  Top 5 net GAINERS:", ', '.join(f"{r.state}({r.net:+d})" for r in top.head(5).itertuples()))
    print("  Top 5 net LOSERS: ", ', '.join(f"{r.state}({r.net:+d})" for r in top.tail(5).itertuples()))

    gdf = get_states_gdf()
    if gdf is not None:
        gdf = attach_abbr(gdf)
    fig9(net_df, gdf)
    fig10(flows, gdf)
    fig_city_bubbles(df, gdf, which='hs', fignum=11)
    fig_city_bubbles(df, gdf, which='birth', fignum=12)
    # Save the migration table too
    net_df.sort_values('net', ascending=False).to_csv('v3_state_net_migration.csv', index=False)
    print("  saved v3_state_net_migration.csv")
    print(f"\nDone. Figures in: {os.path.abspath(OUTDIR)}")


if __name__ == '__main__':
    main()
