"""
Tier value across post-draft WINDOWS, to control for cohort age imbalance.

Whole-career totals overweight older drafts (a 1996 draftee has 25+ seasons to
accumulate; a 2019 draftee ~6). A fixed post-draft window puts every player on
the same clock. But short windows capture mostly rookie-scuffle years for
HS draftees (18 at draft, typical debut 4-6 yrs out), which is noisy for a
RATE stat that wants established samples. So we run THREE views side by side and
check whether the hitter-decline / pitcher-flat split is stable across all:

  - draft+6   (7-season span, draft year inclusive)  -- fairest cross-cohort
  - draft+8   (9-season span)                          -- more established
  - career    (no upper bound, draft year inclusive)   -- most complete

Window is INCLUSIVE of the draft year (Season >= draft_year) to catch the rare
fast HS callup. Data assumed complete through MAX_SEASON (auto-detected from the
season files). A cohort is CENSORED for a window if draft_year + span - 1 >
MAX_SEASON; censored cohorts are excluded from that window's rate/season/WAR
(but always counted in ever-reached).

Metrics per tier x type x window:
  n_reached_ever      -- ever reached MLB (career; not windowed)
  n_reached_in_window -- reached within this window
  mean_seasons        -- distinct MLB seasons within window
  total_war           -- summed WAR within window (no scaling)
  rate_floored        -- WAR/600PA or /150IP, floored 145PA/50IP
  war_per_season      -- window WAR / distinct window seasons

Run:  python scripts/v3_tier_windows.py
Output: v3_tier_windows.csv
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
WINDOWS = [('draft+6', 7), ('draft+8', 9), ('career', None)]  # span incl draft yr
MAX_SEASON_CAP = 2025   # exclude in-progress 2026 (non-reproducible, partial)


def detect_kind(cols):
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


def load_seasons():
    if not (os.path.exists(HIT_SEASONS) and os.path.exists(PIT_SEASONS)):
        print("ERROR: season files required for windowed analysis.")
        sys.exit(1)
    a = pd.read_csv(HIT_SEASONS, low_memory=False)
    b = pd.read_csv(PIT_SEASONS, low_memory=False)
    hit = a if detect_kind(a.columns) == 'HIT' else b
    pit = b if detect_kind(b.columns) == 'PIT' else a
    idc = 'MLBAMID' if 'MLBAMID' in hit.columns else 'PlayerId'
    for d in (hit, pit):
        d['Season'] = pd.to_numeric(d['Season'], errors='coerce')
    # cap out in-progress / partial seasons for reproducibility
    n_before = len(hit) + len(pit)
    hit = hit[hit['Season'] <= MAX_SEASON_CAP]
    pit = pit[pit['Season'] <= MAX_SEASON_CAP]
    n_dropped = n_before - (len(hit) + len(pit))
    if n_dropped:
        print(f"  capped seasons at {MAX_SEASON_CAP}: dropped {n_dropped} "
              f"post-{MAX_SEASON_CAP} season-rows")
    max_season = int(max(hit['Season'].max(), pit['Season'].max()))
    return hit, pit, idc, max_season


def tier(d):
    if d <= SAME_CITY_MAX:
        return '0 Stayer'
    names = ['1 Short', '2 Medium', '3 Long']
    for i, e in enumerate(EDGES):
        if d <= e:
            return names[i]
    return names[len(EDGES)]


def window_agg(hit, pit, idc, draft_year_map, span, max_season):
    """Aggregate hit/pit season rows into per-player window totals.
       span=None => career (no upper bound). Returns df keyed by idc."""
    def agg(df, war, samp):
        d = df.dropna(subset=[idc, 'Season']).copy()
        d['dy'] = d[idc].map(draft_year_map)
        d = d[d['dy'].notna()]
        d = d[d['Season'] >= d['dy']]                     # incl draft year
        if span is not None:
            d = d[d['Season'] <= d['dy'] + span - 1]      # span seasons inclusive
        g = d.groupby(idc).agg(
            war=(war, 'sum'), samp=(samp, 'sum'),
            seasons=('Season', 'nunique')).reset_index()
        return g
    h = agg(hit, 'WAR', 'PA').rename(columns={'war': 'hwar', 'samp': 'pa',
                                              'seasons': 'hs'})
    p = agg(pit, 'WAR', 'IP').rename(columns={'war': 'pwar', 'samp': 'ip',
                                              'seasons': 'ps'})
    m = h.merge(p, on=idc, how='outer')
    for c in ['hwar', 'pa', 'hs', 'pwar', 'ip', 'ps']:
        m[c] = m[c].fillna(0)
    m['seasons'] = m[['hs', 'ps']].max(axis=1)            # two-way union
    m['ctype'] = np.where(m['ip']*4.2 >= m['pa'], 'Pitcher', 'Hitter')
    return m


def main():
    df = load_analysis()
    df['tier'] = df['dist'].apply(tier)
    hit, pit, idc, max_season = load_seasons()
    print(f"Rows {COHORT_LABEL}: {len(df):,}   season data through {max_season}")

    # id to join analysis <-> season files
    acol = next((c for c in ['mlbid', 'MLBAMID', 'mlbid_clean'] if c in df.columns), None)
    if acol is None:
        print("ERROR: no player id column in analysis to join seasons.")
        sys.exit(1)
    df['_id'] = pd.to_numeric(df[acol], errors='coerce')
    draft_year_map = df.dropna(subset=['_id']).set_index('_id')['year'].to_dict()
    tier_map = df.dropna(subset=['_id']).set_index('_id')['tier'].to_dict()

    all_rows = []
    for wlabel, span in WINDOWS:
        wa = window_agg(hit, pit, idc, draft_year_map, span, max_season)
        wa['tier'] = wa[idc].map(tier_map)
        wa['draft_year'] = wa[idc].map(draft_year_map)
        # censor: cohort needs draft_year+span-1 <= max_season
        if span is not None:
            wa['censored'] = (wa['draft_year'] + span - 1 > max_season)
        else:
            wa['censored'] = False
        wa_use = wa[~wa['censored']].copy()
        n_cens = int(wa['censored'].sum())

        # windowed reach = appeared at all in window (>=1 season)
        wa_use = wa_use[wa_use['seasons'] >= 1]

        print("\n" + "="*88)
        cens_note = f"  (censored cohorts excluded: {n_cens} players)" if span else ""
        print(f"WINDOW: {wlabel}{cens_note}")
        print("="*88)

        for typ, warc, sampc, scale, floor, unit in [
                ('Hitter', 'hwar', 'pa', 600, PA_FLOOR, 'PA'),
                ('Pitcher', 'pwar', 'ip', 150, IP_FLOOR, 'IP')]:
            rt = wa_use[wa_use['ctype'] == typ].copy()
            rt['rate'] = np.where(rt[sampc] >= floor, rt[warc]/rt[sampc]*scale, np.nan)
            rt['wps'] = np.where(rt['seasons'] > 0, rt[warc]/rt['seasons'], np.nan)
            print(f"\n  {typ}s (WAR/{scale}{unit}, floor {floor}{unit}):")
            print(f"    {'tier':<9}{'Nwin':>6}{'seas':>6}{'totWAR':>8}"
                  f"{'rate':>8}{'nQ':>5}{'WAR/seas':>9}")
            for t in ORDER:
                ts = rt[rt['tier'] == t]
                if len(ts) < 5:
                    continue
                nq = int(ts['rate'].notna().sum())
                print(f"    {t:<9}{len(ts):>6}{ts['seasons'].mean():>6.1f}"
                      f"{ts[warc].mean():>8.2f}{ts['rate'].mean():>8.2f}{nq:>5}"
                      f"{ts['wps'].mean():>9.2f}")
                all_rows.append({'window': wlabel, 'type': typ, 'tier': t,
                                 'n_window': len(ts),
                                 'mean_seasons': round(ts['seasons'].mean(), 2),
                                 'total_war': round(ts[warc].mean(), 3),
                                 'rate': round(ts['rate'].mean(), 3), 'n_qual': nq,
                                 'war_per_season': round(ts['wps'].mean(), 3)})

    pd.DataFrame(all_rows).to_csv('v3_tier_windows.csv', index=False)
    print("\n" + "="*88)
    print("READ: if hitter rate declines Stayer->Long in ALL THREE windows while")
    print("pitchers stay flat, the finding is window-invariant. If it only appears")
    print("in longer windows, the effect needs time to reveal (a real claim itself).")
    print("Watch nQ on the hitter Long cell -- shorter windows thin it.")
    print("\nSaved: v3_tier_windows.csv")


if __name__ == '__main__':
    main()
