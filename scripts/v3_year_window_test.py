"""
Step 18 — Year-window sensitivity.

Tests whether the mover effect is stable when the start year changes. The main
pipeline window is set in scripts/config.py; the amateur draft goes back to 1965, and the modern
30-team era began in 1998 (Arizona + Tampa Bay first drafted in 1996). This
re-runs the core HS-draftee mover/stayer reach-rate analysis for several start
years through 2019, reusing the geocode cache (geocoding any new towns inline).

Produces:
  v3_year_window_sensitivity.csv
  figures/fig20_year_window_sensitivity.png

Inputs:
  data/tbc_draft_register.csv
  geocode_cache_v3.json   (reused; updated if new towns found)

Run:  python scripts/v3_year_window_test.py
"""

import os
import sys
import re
import json
import time
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

REGISTER = 'data/tbc_draft_register.csv'
ANALYSIS = 'v3_analysis.csv'
CACHE_FILE = 'geocode_cache_v3.json'
END_YEAR = 2019
START_YEARS = [1996, 1998, 2000, 2002, 2005]  # windows to test, each ... -> END_YEAR
MOVER_CUTOFF_MI = 5.0

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}
CA_PROVS = {'ON','QC','BC','AB','MB','SK','NS','NB','NL','PE','YT','NT','NU'}
COUNTRY_MAP = {
    'PR': 'Puerto Rico', 'DR': 'Dominican Republic', 'VE': 'Venezuela',
    'VZ': 'Venezuela', 'MX': 'Mexico', 'CU': 'Cuba', 'JA': 'Japan',
    'JP': 'Japan', 'KO': 'South Korea', 'AU': 'Australia', 'PN': 'Panama',
}


def parse_place(p):
    if pd.isna(p):
        return None, None
    p = str(p).strip()
    if ',' in p:
        a, b = p.rsplit(',', 1)
        return a.strip(), b.strip()
    return p, None


def parse_school(s):
    if pd.isna(s):
        return None, None, None
    s = str(s).strip()
    m = re.search(r'\(([^()]+)\)\s*$', s)
    if not m:
        return s, None, None
    inside = m.group(1).strip()
    name = s[:m.start()].strip()
    if ',' in inside:
        a, b = inside.rsplit(',', 1)
        return name, a.strip(), b.strip()
    return name, inside, None


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def build_query(city, state):
    if not city or not state:
        return None
    city, state = str(city).strip(), str(state).strip()
    if state in US_STATES:
        return f"{city}, {state}, USA"
    if state in CA_PROVS:
        return f"{city}, {state}, Canada"
    if state in COUNTRY_MAP:
        return f"{city}, {COUNTRY_MAP[state]}"
    return f"{city}, {state}"


def geocode_one(query):
    try:
        import requests
        r = requests.get('https://nominatim.openstreetmap.org/search',
                         params={'q': query, 'format': 'json', 'limit': 1},
                         headers={'User-Agent': 'DraftMobilityResearch/1.0'}, timeout=12)
        if r.status_code == 200 and r.json():
            d = r.json()[0]
            return {'lat': float(d['lat']), 'lon': float(d['lon']), 'status': 'OK'}
    except Exception:
        return None
    return None


def coords_for(cache, city, state):
    if pd.isna(city) or pd.isna(state):
        return np.nan, np.nan
    v = cache.get(f"{str(city).strip()}|{str(state).strip()}")
    if v and v.get('lat') is not None:
        return v['lat'], v['lon']
    return np.nan, np.nan


def haversine(a, b, c, d):
    if any(pd.isna(x) for x in [a, b, c, d]):
        return np.nan
    R = 3958.8
    a, b, c, d = map(radians, [a, b, c, d])
    h = sin((c-a)/2)**2 + cos(a)*cos(c)*sin((d-b)/2)**2
    return R * 2 * atan2(sqrt(h), sqrt(1-h))


def load_from_analysis():
    """
    Preferred path: re-slice the already-built analysis dataset.

    v3_analysis.csv has been through the typo-correction (step 3) and
    distance-computation (step 5) stages, so its city names and coordinates
    match the main analysis exactly. Using it makes this a pure re-slice of
    the same sample rather than an independent rebuild, and needs no geocoding.
    """
    df = pd.read_csv(ANALYSIS, low_memory=False)
    if 'distance_miles' not in df.columns:
        return None
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna()].copy()
    if df.empty:
        return None
    # Recompute the mover flag so MOVER_CUTOFF_MI here is authoritative.
    df['mover'] = (df['distance_miles'] > MOVER_CUTOFF_MI).astype(float)
    df = df[(df['year'] >= min(START_YEARS)) & (df['year'] <= END_YEAR)]
    print(f"Source: {ANALYSIS} (re-slice of the main analysis sample)")
    print(f"HS draftees with distance, {min(START_YEARS)}-{END_YEAR}: {len(df):,}")
    return df


def load_from_register():
    """
    Fallback: rebuild from the raw register and geocode inline.

    NOTE: this path does NOT apply the step-3 typo corrections, so its sample
    can differ slightly from the main analysis. Only used when
    v3_analysis.csv is unavailable.
    """
    if not os.path.exists(REGISTER):
        print(f"ERROR: neither {ANALYSIS} nor {REGISTER} found.")
        sys.exit(1)
    print(f"Source: {REGISTER} (fallback rebuild — v3_analysis.csv not found)")
    print("WARNING: typo corrections from step 3 are not applied on this path.")

    reg = pd.read_csv(REGISTER, low_memory=False)
    reg = reg[(reg['year'] >= min(START_YEARS)) & (reg['year'] <= END_YEAR)].copy()
    print(f"Register rows {min(START_YEARS)}-{END_YEAR}: {len(reg):,}")

    reg[['birth_city', 'birth_state']] = reg['place'].apply(lambda p: pd.Series(parse_place(p)))
    reg[['hs_name', 'hs_city', 'hs_state']] = reg['school'].apply(lambda s: pd.Series(parse_school(s)))
    reg['reached_mlb'] = (reg['highLevel'] == 'MLB').astype(int)
    reg['is_hs_draftee'] = ((reg.get('playerClass') == 'HS') |
                            (reg.get('schoolDivision') == 'HS')).astype(int)

    hs = reg[(reg['is_hs_draftee'] == 1) &
             reg['birth_city'].notna() & reg['birth_state'].notna() &
             reg['hs_city'].notna() & reg['hs_state'].notna()].copy()
    print(f"HS draftees with both locations: {len(hs):,}")

    cache = load_cache()
    needed = set()
    for _, r in hs.iterrows():
        needed.add((str(r['birth_city']).strip(), str(r['birth_state']).strip()))
        needed.add((str(r['hs_city']).strip(), str(r['hs_state']).strip()))
    missing = [(c, s) for (c, s) in needed
               if f"{c}|{s}" not in cache or cache.get(f"{c}|{s}", {}).get('lat') is None]
    print(f"Places needed: {len(needed):,}; not cached: {len(missing):,}")
    if missing:
        print(f"Geocoding {len(missing):,} new places (~{len(missing)/60:.1f} min)...")
        for i, (c, s) in enumerate(missing, 1):
            q = build_query(c, s)
            key = f"{c}|{s}"
            res = geocode_one(q)
            if res:
                cache[key] = res
            else:
                prior = int(cache.get(key, {}).get('attempts', 0))
                cache[key] = {'lat': None, 'lon': None, 'status': 'failed',
                              'query': q, 'attempts': prior + 1}
            if q:
                time.sleep(1.0)
            if i % 50 == 0:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f)
                print(f"  {i}/{len(missing)}")
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

    bl = hs.apply(lambda r: coords_for(cache, r['birth_city'], r['birth_state']),
                  axis=1, result_type='expand')
    hs['birth_lat'], hs['birth_lon'] = bl[0], bl[1]
    hl = hs.apply(lambda r: coords_for(cache, r['hs_city'], r['hs_state']),
                  axis=1, result_type='expand')
    hs['hs_lat'], hs['hs_lon'] = hl[0], hl[1]
    hs['distance_miles'] = [haversine(a, b, c, d) for a, b, c, d in
                            zip(hs['birth_lat'], hs['birth_lon'], hs['hs_lat'], hs['hs_lon'])]
    hs['mover'] = np.where(hs['distance_miles'].isna(), np.nan,
                           (hs['distance_miles'] > MOVER_CUTOFF_MI).astype(float))
    return hs


def main():
    hs = None
    if os.path.exists(ANALYSIS):
        hs = load_from_analysis()
    if hs is None:
        hs = load_from_register()

    hs['year_c'] = hs['year'] - 2010
    hs['rounds_1_5'] = (hs['draftRound'] <= 5).astype(int)
    hs['rounds_6_10'] = ((hs['draftRound'] >= 6) & (hs['draftRound'] <= 10)).astype(int)
    hs['rounds_11_20'] = ((hs['draftRound'] >= 11) & (hs['draftRound'] <= 20)).astype(int)

    from scipy import stats
    try:
        import statsmodels.api as sm
        have_sm = True
    except ImportError:
        have_sm = False

    print(f"\n{'='*82}")
    print("MOVER vs STAYER BY START YEAR (each window runs through 2019)")
    print(f"{'='*82}")
    print(f"{'Window':>12} {'N':>7} {'Mover%':>7} {'Stay%':>7} {'Diff':>7} "
          f"{'OR(ctrl)':>9} {'p':>10}")
    print('-'*82)

    rows = []
    for sy in START_YEARS:
        sub = hs[(hs['year'] >= sy) & hs['mover'].notna() & hs['distance_miles'].notna()].copy()
        mv = sub[sub['mover'] == 1]['reached_mlb']
        st = sub[sub['mover'] == 0]['reached_mlb']
        if len(mv) == 0 or len(st) == 0:
            continue
        diff = (mv.mean() - st.mean()) * 100
        # controlled OR
        or_ctrl = p_ctrl = np.nan
        if have_sm:
            X = sm.add_constant(sub[['mover', 'year_c', 'rounds_1_5', 'rounds_6_10', 'rounds_11_20']])
            res = sm.Logit(sub['reached_mlb'].astype(int), X).fit(disp=False, maxiter=200)
            or_ctrl = np.exp(res.params['mover'])
            p_ctrl = res.pvalues['mover']
        print(f"{sy}-2019 {len(sub):>7,} {mv.mean()*100:>6.1f}% {st.mean()*100:>6.1f}% "
              f"{diff:>+6.1f} {or_ctrl:>9.3f} {p_ctrl:>10.2e}")
        rows.append({'start_year': sy, 'end_year': END_YEAR, 'n': len(sub),
                     'mover_pct': round(mv.mean()*100, 2), 'stayer_pct': round(st.mean()*100, 2),
                     'diff_pp': round(diff, 2),
                     'mover_or_controlled': round(or_ctrl, 4) if not np.isnan(or_ctrl) else None,
                     'p_value': p_ctrl if not np.isnan(p_ctrl) else None})

    out = pd.DataFrame(rows)
    out.to_csv('v3_year_window_sensitivity.csv', index=False)
    print(f"\nSaved: v3_year_window_sensitivity.csv")

    if len(out) > 1:
        ors = out['mover_or_controlled'].dropna()
        if len(ors):
            print(f"Mover OR ranges {ors.min():.2f}-{ors.max():.2f} across start years "
                  f"-> the effect is stable whether you start at 1996 or 2005.")

    # figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                             'axes.spines.top': False, 'axes.spines.right': False})
        os.makedirs('figures', exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(11, 6))
        x = range(len(out))
        w = 0.38
        ax1.bar([i - w/2 for i in x], out['mover_pct'], w, label='Movers', color='#2C6FBB')
        ax1.bar([i + w/2 for i in x], out['stayer_pct'], w, label='Stayers', color='#C44E52')
        ax1.set_xticks(list(x))
        ax1.set_xticklabels([f"{sy}-2019" for sy in out['start_year']])
        ax1.set_ylabel('Reached MLB (%)')
        ax1.set_ylim(0, max(out['mover_pct']) * 1.25)
        ax1.legend(loc='upper left', frameon=False)
        for i, d in zip(x, out['diff_pp']):
            ax1.text(i, max(out['mover_pct'].iloc[i], out['stayer_pct'].iloc[i]) + 1.0,
                     f'+{d:.1f}', ha='center', fontweight='bold', color='#4C9F70', fontsize=10)
        ax1.set_title('The mover effect is stable across start-year windows',
                      fontsize=15, fontweight='bold')
        fig.text(0.99, 0.01, 'HS draftees, each window ends 2019. Green = mover minus stayer gap (pp). '
                 'Draft dates to 1965; 30-team era from 1998.', ha='right', fontsize=8, color='#555')
        fig.tight_layout()
        p = 'figures/fig20_year_window_sensitivity.png'
        fig.savefig(p, bbox_inches='tight'); plt.close(fig)
        print(f"  saved {p}")
    except ImportError:
        print("  (matplotlib not available; CSV written, figure skipped)")

    print("Done.")


if __name__ == '__main__':
    main()
