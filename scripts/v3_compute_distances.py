"""
v3_compute_distances.py

Joins geocoded coordinates to v3_analysis.csv and computes:
  - distance_miles:    haversine distance from birth city to HS city
  - log_distance:      log(distance_miles + 1)
  - mover:             1 if distance_miles > 5, 0 if <= 5, NaN if missing
  - distance_bin:      categorical bin label
  - same_city:         1 if birth city name == school city name (exact)
  - changed_state:     1 if birth state != school state
  - is_foreign_born:   1 if birth_state not in US states / Canadian provinces
  - birth_warm_state:  1 if born in a warm-weather baseball state
  - hs_warm_state:     1 if HS in a warm-weather baseball state
  - hs_city:           alias for school_city (for downstream scripts)
  - hs_state:          alias for school_state (for downstream scripts)
  - hs_name:           alias for school_name (for downstream scripts)

Input:
  v3_analysis.csv        (from build_v3_analysis.py)
  city_coords_v3.csv     (from v3_geocode.py / fix_and_refill_geocodes.py)

Output:
  v3_analysis.csv        (updated in place with new columns)
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ── geography constants ──────────────────────────────────────────────────────
US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}
CA_PROVS = {'ON','QC','BC','AB','MB','SK','NS','NB','NL','PE','YT','NT','NU'}
DOMESTIC = US_STATES | CA_PROVS | {'PR'}   # PR treated as domestic

# Warm-weather US states where year-round outdoor baseball is possible
WARM_STATES = {'FL','TX','CA','AZ','GA','NC','SC','AL','MS','LA','NV','HI','PR'}

# Distance bins (matches labels used in v3_war_analysis.py and v3_cohort_2000_2019.py)
BIN_EDGES  = [0, 0.1, 5, 10, 15, 20, 25, 50, 100, 250, 500, 1000, float('inf')]
BIN_LABELS = ['00_exact', '01_under5', '02_5_10', '03_10_15', '04_15_20',
              '05_20_25', '06_25_50', '07_50_100', '08_100_250',
              '09_250_500', '10_500_1000', '11_1000_plus']

def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def assign_bin(dist):
    if pd.isna(dist):
        return np.nan
    for i, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
        if lo <= dist < hi:
            return BIN_LABELS[i]
    return BIN_LABELS[-1]

# ── load data ────────────────────────────────────────────────────────────────
print("Loading v3_analysis.csv ...")
df = pd.read_csv('v3_analysis.csv', low_memory=False)
print(f"  {len(df)} rows, {len(df.columns)} columns")

print("Loading city_coords_v3.csv ...")
coords = pd.read_csv('city_coords_v3.csv', low_memory=False)
# Keep only successfully geocoded rows
coords = coords[coords['lat'].notna() & coords['lon'].notna()].copy()
print(f"  {len(coords)} geocoded (city, state) pairs")

# Build a fast lookup dict: "City|ST" -> (lat, lon)
coord_dict = {}
for _, r in coords.iterrows():
    key = f"{str(r['city']).strip()}|{str(r['state']).strip()}"
    coord_dict[key] = (float(r['lat']), float(r['lon']))

def get_coords(city, state):
    if pd.isna(city) or pd.isna(state):
        return np.nan, np.nan
    key = f"{str(city).strip()}|{str(state).strip()}"
    if key in coord_dict:
        return coord_dict[key]
    return np.nan, np.nan

# ── geocode birth and school cities ─────────────────────────────────────────
print("Geocoding birth cities ...")
birth_lats, birth_lons = [], []
for _, r in df.iterrows():
    la, lo = get_coords(r.get('birth_city'), r.get('birth_state'))
    birth_lats.append(la)
    birth_lons.append(lo)
df['birth_lat'] = birth_lats
df['birth_lon'] = birth_lons

print("Geocoding school (HS) cities ...")
hs_lats, hs_lons = [], []
hs_city_col   = 'school_city'   if 'school_city'   in df.columns else 'hs_city'
hs_state_col  = 'school_state'  if 'school_state'  in df.columns else 'hs_state'
for _, r in df.iterrows():
    la, lo = get_coords(r.get(hs_city_col), r.get(hs_state_col))
    hs_lats.append(la)
    hs_lons.append(lo)
df['hs_lat'] = hs_lats
df['hs_lon'] = hs_lons

# ── compute distance ─────────────────────────────────────────────────────────
print("Computing haversine distances ...")
df['distance_miles'] = [
    haversine(blat, blon, hlat, hlon)
    for blat, blon, hlat, hlon
    in zip(df['birth_lat'], df['birth_lon'], df['hs_lat'], df['hs_lon'])
]
df['log_distance'] = np.log1p(df['distance_miles'])

# ── mover flag ───────────────────────────────────────────────────────────────
df['mover'] = np.where(
    df['distance_miles'].isna(), np.nan,
    (df['distance_miles'] > 5).astype(float)
)

# ── distance bin ─────────────────────────────────────────────────────────────
df['distance_bin'] = df['distance_miles'].apply(assign_bin)

# ── same city (exact name match) ─────────────────────────────────────────────
df['same_city'] = (
    df['birth_city'].str.lower().str.strip() ==
    df[hs_city_col].str.lower().str.strip()
).astype('Int64')   # nullable int so NaN rows stay NaN
# Set to NaN where either city is missing
mask_missing = df['birth_city'].isna() | df[hs_city_col].isna()
df.loc[mask_missing, 'same_city'] = pd.NA

# ── changed state ─────────────────────────────────────────────────────────────
df['changed_state'] = (
    df['birth_state'].str.upper().str.strip() !=
    df[hs_state_col].str.upper().str.strip()
).astype('Int64')
mask_st = df['birth_state'].isna() | df[hs_state_col].isna()
df.loc[mask_st, 'changed_state'] = pd.NA

# ── foreign-born flag ─────────────────────────────────────────────────────────
df['is_foreign_born'] = df['birth_state'].apply(
    lambda s: 0 if (pd.notna(s) and str(s).strip().upper() in DOMESTIC) else
              (1 if pd.notna(s) else np.nan)
).astype('Int64')

# ── warm-state flags ─────────────────────────────────────────────────────────
df['birth_warm_state'] = df['birth_state'].apply(
    lambda s: int(str(s).strip().upper() in WARM_STATES) if pd.notna(s) else np.nan
)
df['hs_warm_state'] = df[hs_state_col].apply(
    lambda s: int(str(s).strip().upper() in WARM_STATES) if pd.notna(s) else np.nan
)

# ── convenient aliases for downstream scripts ─────────────────────────────────
if 'hs_city' not in df.columns:
    df['hs_city']  = df[hs_city_col]
if 'hs_state' not in df.columns:
    df['hs_state'] = df[hs_state_col]
if 'hs_name' not in df.columns:
    df['hs_name']  = df.get('school_name', np.nan)

# ── coverage report ──────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("DISTANCE COVERAGE")
print(f"{'='*65}")
has_dist = df['distance_miles'].notna()
hs_mask  = (df['is_hs_draftee'] == 1) if 'is_hs_draftee' in df.columns else pd.Series([True]*len(df))
hs = df[hs_mask]
print(f"All draftees:     {len(df):6d}   with distance: {has_dist.sum():6d} ({has_dist.mean()*100:.1f}%)")
hs_dist = hs['distance_miles'].notna()
print(f"HS draftees only: {len(hs):6d}   with distance: {hs_dist.sum():6d} ({hs_dist.mean()*100:.1f}%)")

print(f"\nHS draftees with distance — mover/stayer split:")
hs_valid = hs[hs['distance_miles'].notna()]
movers  = hs_valid[hs_valid['mover'] == 1]
stayers = hs_valid[hs_valid['mover'] == 0]
print(f"  Movers  (>5 mi):  {len(movers):5d} ({len(movers)/len(hs_valid)*100:.1f}%)")
print(f"  Stayers (≤5 mi):  {len(stayers):5d} ({len(stayers)/len(hs_valid)*100:.1f}%)")

if 'reached_mlb' in df.columns:
    print(f"\nMLB reach rates (HS draftees with distance):")
    print(f"  Movers:  {movers['reached_mlb'].mean()*100:.1f}%  (n={len(movers)})")
    print(f"  Stayers: {stayers['reached_mlb'].mean()*100:.1f}%  (n={len(stayers)})")
    from scipy import stats
    ct = pd.crosstab(hs_valid['mover'], hs_valid['reached_mlb'])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    print(f"  Chi-squared: {chi2:.2f}, p = {p:.8f}")

print(f"\nDistance distribution (HS draftees):")
print(hs_valid['distance_miles'].describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(1).to_string())

print(f"\nDistance bin counts (HS draftees):")
print(hs_valid['distance_bin'].value_counts().sort_index().to_string())

# ── save ─────────────────────────────────────────────────────────────────────
df.to_csv('v3_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved: v3_analysis.csv  ({len(df)} rows, {len(df.columns)} columns)")
print("New columns added:")
new_cols = ['birth_lat','birth_lon','hs_lat','hs_lon','distance_miles','log_distance',
            'mover','distance_bin','same_city','changed_state','is_foreign_born',
            'birth_warm_state','hs_warm_state','hs_city','hs_state','hs_name']
for c in new_cols:
    if c in df.columns:
        nn = df[c].notna().sum()
        print(f"  {c:<22} {nn:>6} non-null ({nn/len(df)*100:.1f}%)")
