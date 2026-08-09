"""
Comprehensive tier value table -- answers all four questions in one view,
excluding NOBODY from the counts / total-WAR columns.

Per tier (and split hitter/pitcher), for players who REACHED MLB:
  1. N reached                          (no threshold)
  2. mean distinct MLB seasons          (from FanGraphs season files, career)
  3. mean career WAR   [NO scaling]     (no threshold -- "total expected WAR")
  4. WAR/600 PA or /150 IP  [FLOORED]   (>=145 PA / >=50 IP; the clean rate)
  5. WAR/600 PA or /150 IP  [NO FLOOR]  (all reachers; robustness / noise check)
  6. WAR per SEASON played              (career WAR / distinct seasons)

Only column 4 applies a qualification floor, and only because a per-600
extrapolation from a tiny sample is noise. Columns 1,2,3,5,6 include every
player who reached MLB. Column 5 sits next to 4 so you can see whether the
floor changes the story (if 4 and 5 agree, bulletproof; if they diverge, the
short-career players behave differently and you want to know).

Distinct seasons reuse the season-file logic from v3_window_rates.py
(detect_kind for the naming quirk, Season.nunique, two-way union) but WITHOUT
the window filter -- these are full-career season counts.

Run:  python scripts/v3_tier_full_value.py
Output: v3_tier_full_value.csv
"""

import os
import sys
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

SAME_CITY_MAX = 1.0
EDGES = [50, 500]
ORDER = ['0 Stayer', '1 Short', '2 Medium', '3 Long']
PA_FLOOR = 145
IP_FLOOR = 50
HIT_SEASONS = 'fg-hit-seasons.csv'
PIT_SEASONS = 'fg-pit-seasons.csv'
MAX_SEASON_CAP = 2025   # exclude in-progress 2026 (non-reproducible, partial)
PITCHER_POS = {'P', 'RHP', 'LHP', 'PITCHER', 'SP', 'RP'}


def is_pitcher_pos(p):
    if p is None:
        return None
    s = str(p).upper().strip()
    if s in ('', 'NAN', 'NONE'):
        return None
    if s in PITCHER_POS or s == 'P' or s.startswith('RHP') or s.startswith('LHP') \
       or s.startswith('P/') or s.startswith('P-') or s.endswith('HP'):
        return True
    return False


def detect_kind(cols):
    """Which season file is which, by CONTENT (immune to the naming swap)."""
    cl = set(c.upper() for c in cols)
    if 'ERA' in cl or 'IP' in cl:
        return 'PIT'
    if 'PA' in cl or 'WOBA' in cl or 'WRC+' in cl:
        return 'HIT'
    return 'HIT'


def load_analysis():
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
    dcol = next((c for c in ['distance_mi', 'distance_miles', 'dist_mi', 'distance']
                 if c in df.columns), None)
    df = df[df[dcol].notna()].rename(columns={dcol: 'dist'})
    return df


def career_seasons():
    """Distinct MLB seasons per player id, career (no window). Returns dict id->count."""
    if not (os.path.exists(HIT_SEASONS) and os.path.exists(PIT_SEASONS)):
        print("  NOTE: season files not found -- distinct-season column will be blank.")
        return {}, None
    a = pd.read_csv(HIT_SEASONS, low_memory=False)
    b = pd.read_csv(PIT_SEASONS, low_memory=False)
    hit = a if detect_kind(a.columns) == 'HIT' else b
    pit = b if detect_kind(b.columns) == 'PIT' else a
    idc = 'MLBAMID' if 'MLBAMID' in hit.columns else 'PlayerId'
    for d in (hit, pit):
        d['Season'] = pd.to_numeric(d['Season'], errors='coerce')
    # cap out in-progress / partial seasons for reproducibility
    hit = hit[hit['Season'] <= MAX_SEASON_CAP]
    pit = pit[pit['Season'] <= MAX_SEASON_CAP]
    print(f"  seasons capped at {MAX_SEASON_CAP} (2026 partial excluded)")
    hs = hit.dropna(subset=[idc, 'Season']).groupby(idc)['Season'].nunique()
    ps = pit.dropna(subset=[idc, 'Season']).groupby(idc)['Season'].nunique()
    both = pd.concat([hs, ps], axis=1).fillna(0)
    both.columns = ['h', 'p']
    seasons = both.max(axis=1).astype(int)   # two-way union
    return seasons.to_dict(), idc


def tier(d):
    if d <= SAME_CITY_MAX:
        return '0 Stayer'
    names = ['1 Short', '2 Medium', '3 Long']
    for i, e in enumerate(EDGES):
        if d <= e:
            return names[i]
    return names[len(EDGES)]


def main():
    df = load_analysis()
    df['tier'] = df['dist'].apply(tier)
    print(f"Rows {COHORT_LABEL}: {len(df):,}   reached MLB: {int(df['reached_mlb'].sum()):,}")

    # distinct career seasons
    seasons_map, sid = career_seasons()
    # map onto analysis rows via the player id the season files use
    idcol = None
    for c in ['mlbid', 'MLBAMID', 'mlbid_clean', 'PlayerID']:
        if c in df.columns:
            idcol = c
            break
    if seasons_map and idcol:
        key = pd.to_numeric(df[idcol], errors='coerce')
        df['n_seasons'] = key.map(seasons_map)
        print(f"  distinct-season match: "
              f"{df['n_seasons'].notna().sum()}/{int(df['reached_mlb'].sum())} reachers")
    else:
        df['n_seasons'] = np.nan

    r = df[df['reached_mlb'] == 1].copy()
    r['pa'] = r['hit_pa'] if 'hit_pa' in r.columns else np.nan
    r['ip'] = r['pit_ip'] if 'pit_ip' in r.columns else np.nan
    r['hwar'] = r['hit_war'] if 'hit_war' in r.columns else r['career_war']
    r['pwar'] = r['pit_war'] if 'pit_war' in r.columns else r['career_war']
    r['ctype'] = np.where(r['ip'].fillna(0)*4.2 >= r['pa'].fillna(0),
                          'Pitcher', 'Hitter')

    rows = []
    for typ, war_col, samp_col, scale, floor in [
            ('Hitter', 'hwar', 'pa', 600, PA_FLOOR),
            ('Pitcher', 'pwar', 'ip', 150, IP_FLOOR)]:
        rt = r[r['ctype'] == typ].copy()
        rt['rate_all'] = rt[war_col] / rt[samp_col] * scale       # no floor
        rt['rate_flr'] = np.where(rt[samp_col] >= floor,
                                  rt[war_col] / rt[samp_col] * scale, np.nan)
        rt['war_per_season'] = np.where(rt['n_seasons'] > 0,
                                        rt['career_war'] / rt['n_seasons'], np.nan)
        unit = 'PA' if typ == 'Hitter' else 'IP'
        print("\n" + "="*84)
        print(f"{typ}s  (rate = WAR/{scale}{unit};  floor = {floor} {unit})")
        print("="*84)
        print(f"  {'tier':<9}{'Nreach':>7}{'seas':>6}{'totWAR':>8}"
              f"{'rate_flr':>9}{'nQ':>5}{'rate_all':>9}{'WAR/seas':>9}")
        for t in ORDER:
            ts = rt[rt['tier'] == t]
            if len(ts) < 5:
                continue
            nq = int(ts['rate_flr'].notna().sum())
            print(f"  {t:<9}{len(ts):>7}{ts['n_seasons'].mean():>6.1f}"
                  f"{ts['career_war'].mean():>8.2f}"
                  f"{ts['rate_flr'].mean():>9.2f}{nq:>5}"
                  f"{ts['rate_all'].mean():>9.2f}"
                  f"{ts['war_per_season'].mean():>9.2f}")
            rows.append({'type': typ, 'tier': t, 'n_reached': len(ts),
                         'mean_seasons': round(ts['n_seasons'].mean(), 2),
                         'total_war': round(ts['career_war'].mean(), 3),
                         'rate_floored': round(ts['rate_flr'].mean(), 3),
                         'n_qual': nq,
                         'rate_nofloor': round(ts['rate_all'].mean(), 3),
                         'war_per_season': round(ts['war_per_season'].mean(), 3)})

    pd.DataFrame(rows).to_csv('v3_tier_full_value.csv', index=False)
    print("\n" + "="*84)
    print("COLUMNS: Nreach & totWAR & seas & WAR/seas exclude nobody who reached.")
    print("rate_flr = floored rate (clean); rate_all = same rate over ALL reachers")
    print("(noisy; if it agrees with rate_flr the finding is robust to the floor).")
    print("\nSaved: v3_tier_full_value.csv")


if __name__ == '__main__':
    main()
