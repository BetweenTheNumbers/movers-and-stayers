"""
12-year post-draft WINDOW rate analysis.

Puts every player on the same clock: seasons (draft_year+1 .. draft_year+12),
excluding the draft year itself. A 2005 pick is measured on 2006-2017, a 2010
pick on 2011-2022. This removes the bias where older draft classes accumulate
more career WAR simply by having more time.

Window handling (per user choice):
  - Everyone is computed; players whose window extends past the latest data
    season are flagged window_censored=1.
  - Headline comparison runs on the UNCENSORED set (full 12 years observed).
  - WAR/year uses a FIXED denominator of 12 (absent years count as zero) --
    this blends quality and durability into "value delivered per year over the
    window". A per-season-PLAYED version is reported alongside for contrast.

Player type + qualification as before (145 PA ~= 130 AB, or 50 IP), but applied
to the IN-WINDOW totals.

Outputs:
  v3_window12_rates.csv                 per player, windowed
  v3_window12_comparison_by_mover.csv   mover vs stayer, uncensored + censored

Run:  python scripts/v3_window_rates.py
"""

import os
import sys
import numpy as np
import pandas as pd

HIT_SEASONS = 'fg-hit-seasons.csv'
PIT_SEASONS = 'fg-pit-seasons.csv'
ANALYSIS = 'v3_analysis_with_war.csv'
OUT_RATES = 'v3_window12_rates.csv'
OUT_CMP = 'v3_window12_comparison_by_mover.csv'

WINDOW = 12
PA_QUAL = 145
IP_QUAL = 50


def _win_arg():
    """Allow --window N (default 12). Also names outputs by window length."""
    import sys
    w = 12
    if '--window' in sys.argv:
        try:
            w = int(sys.argv[sys.argv.index('--window') + 1])
        except (IndexError, ValueError):
            pass
    return w


def detect_kind(cols):
    cl = {c.lower() for c in cols}
    if {'era', 'fip', 'ip'} & cl:
        return 'PIT'
    if {'woba', 'wrc+', 'pa', 'obp'} & cl:
        return 'HIT'
    return '?'


def load(fname):
    if not os.path.exists(fname):
        print(f"ERROR: {fname} not found.")
        sys.exit(1)
    return pd.read_csv(fname, low_memory=False)


def main():
    global WINDOW, OUT_RATES, OUT_CMP
    WINDOW = _win_arg()
    OUT_RATES = f'v3_window{WINDOW}_rates.csv'
    OUT_CMP = f'v3_window{WINDOW}_comparison_by_mover.csv'
    print(f"Window length: {WINDOW} years (draft+1 .. draft+{WINDOW})")

    a = load(HIT_SEASONS)
    b = load(PIT_SEASONS)
    hit = a if detect_kind(a.columns) == 'HIT' else b
    pit = b if detect_kind(b.columns) == 'PIT' else a
    idc = 'MLBAMID' if 'MLBAMID' in hit.columns else 'PlayerId'
    for d in (hit, pit):
        d[idc] = pd.to_numeric(d[idc], errors='coerce')
        d['Season'] = pd.to_numeric(d['Season'], errors='coerce')
    max_season = int(max(hit['Season'].max(), pit['Season'].max()))
    print(f"Latest season in data: {max_season}")

    # ---- need draft year per player: from analysis file ------------------
    if not os.path.exists(ANALYSIS):
        print(f"ERROR: {ANALYSIS} not found.")
        sys.exit(1)
    an = pd.read_csv(ANALYSIS, low_memory=False)
    an = an[(an.get('is_hs_draftee', 0) == 1) & (an['reached_mlb'] == 1)].copy()
    idcol = 'mlbid_clean' if 'mlbid_clean' in an.columns else 'mlbid'
    an[idcol] = pd.to_numeric(an[idcol], errors='coerce')
    # earliest draft year per player (a player can be drafted more than once)
    draft_year = an.groupby(idcol)['year'].min().to_dict()
    mover_map = (an.sort_values('year').drop_duplicates(idcol, keep='first')
                 .set_index(idcol)['mover'].to_dict())

    def window_sum(df, war_col='WAR'):
        """Sum stats within each player's (draft+1 .. draft+12) window."""
        df = df.dropna(subset=[idc, 'Season']).copy()
        df['dy'] = df[idc].map(draft_year)
        df = df[df['dy'].notna()]
        df = df[(df['Season'] >= df['dy'] + 1) &
                (df['Season'] <= df['dy'] + WINDOW)]
        return df

    hw = window_sum(hit)
    pw = window_sum(pit)

    hg = hw.groupby(idc).agg(
        hit_war=('WAR', 'sum'), hit_pa=('PA', 'sum'), hit_g=('G', 'sum'),
        hit_seasons=('Season', 'nunique')).reset_index()
    pg = pw.groupby(idc).agg(
        pit_war=('WAR', 'sum'), pit_ip=('IP', 'sum'), pit_g=('G', 'sum'),
        pit_seasons=('Season', 'nunique')).reset_index()

    p = hg.merge(pg, on=idc, how='outer')
    for c in ['hit_war', 'hit_pa', 'hit_g', 'hit_seasons',
              'pit_war', 'pit_ip', 'pit_g', 'pit_seasons']:
        p[c] = p[c].fillna(0)
    p['win_war'] = p['hit_war'] + p['pit_war']
    p['win_g'] = p['hit_g'] + p['pit_g']
    # seasons appeared in-window (union; use max since two-way overlap)
    p['seasons_in_window'] = p[['hit_seasons', 'pit_seasons']].max(axis=1)

    p['draft_year'] = p[idc].map(draft_year)
    p['mover'] = p[idc].map(mover_map)
    p['window_censored'] = (p['draft_year'] + WINDOW > max_season).astype(int)

    # type + qualification (in-window)
    def ptype(r):
        h = r['hit_pa'] >= PA_QUAL
        pi = r['pit_ip'] >= IP_QUAL
        if h and pi:
            return 'Two-Way'
        if pi and not h:
            return 'Pitcher'
        if h and not pi:
            return 'Hitter'
        return 'Pitcher' if r['pit_ip'] * 4.3 >= r['hit_pa'] else 'Hitter'
    p['player_type'] = p.apply(ptype, axis=1)
    p['qualified_hit'] = (p['hit_pa'] >= PA_QUAL).astype(int)
    p['qualified_pit'] = (p['pit_ip'] >= IP_QUAL).astype(int)
    p['qualified'] = ((p['qualified_hit'] == 1) | (p['qualified_pit'] == 1)).astype(int)

    # rates
    def safe(n, d):
        return np.where(d > 0, n / np.where(d == 0, np.nan, d), np.nan)
    p['war12_per_year'] = p['win_war'] / WINDOW                    # fixed denom
    p['war12_per_season_played'] = safe(p['win_war'], p['seasons_in_window'])
    p['war12_per_150g'] = safe(p['win_war'], p['win_g']) * 150
    p['war12_per_600pa'] = safe(p['hit_war'], p['hit_pa']) * 600
    p['war12_per_150ip'] = safe(p['pit_war'], p['pit_ip']) * 150
    for c in ['war12_per_season_played', 'war12_per_150g']:
        p.loc[p['qualified'] == 0, c] = np.nan
    p.loc[p['qualified_hit'] == 0, 'war12_per_600pa'] = np.nan
    p.loc[p['qualified_pit'] == 0, 'war12_per_150ip'] = np.nan

    p = p.rename(columns={idc: 'MLBAMID'})
    p.to_csv(OUT_RATES, index=False)
    print(f"\nSaved {OUT_RATES}: {len(p):,} players")
    print(f"  Uncensored: {(p.window_censored==0).sum():,}   "
          f"Censored: {(p.window_censored==1).sum():,}")
    print(f"  Hitters {(p.player_type=='Hitter').sum():,} / "
          f"Pitchers {(p.player_type=='Pitcher').sum():,} / "
          f"Two-Way {(p.player_type=='Two-Way').sum():,}")

    # ---- comparison ------------------------------------------------------
    from scipy import stats
    rows = []

    def block(df, tag):
        print(f"\n{'='*82}\n{tag}  (n={len(df):,})\n{'='*82}")

        def cmp(sub, col, label, need=None):
            s = sub if need is None else sub[sub[need] == 1]
            s = s[s[col].notna()]
            mv, st = s[s.mover == 1][col], s[s.mover == 0][col]
            if len(mv) < 10 or len(st) < 10:
                return
            t, pv = stats.ttest_ind(mv, st, equal_var=False)
            star = '  <-- differs' if pv < 0.05 else ''
            print(f"  {label:<34} movers {mv.mean():7.3f} (n={len(mv):4d})  "
                  f"stayers {st.mean():7.3f} (n={len(st):4d})  "
                  f"diff {mv.mean()-st.mean():+6.3f}  p={pv:.4f}{star}")
            rows.append({'sample': tag, 'group': label, 'metric': col,
                         'mover_mean': round(mv.mean(), 4), 'mover_n': len(mv),
                         'stayer_mean': round(st.mean(), 4), 'stayer_n': len(st),
                         'diff': round(mv.mean()-st.mean(), 4), 'p_value': pv})

        q = df[df.qualified == 1]
        cmp(q, 'war12_per_year', f'ALL war{WINDOW}/yr (fixed /{WINDOW})')
        cmp(q, 'war12_per_season_played', f'ALL war{WINDOW}/season played')
        cmp(q, 'seasons_in_window', f'ALL seasons in window (0-{WINDOW})')
        cmp(df[df.player_type == 'Hitter'], 'war12_per_600pa', f'HIT war{WINDOW}/600pa', 'qualified_hit')
        cmp(df[df.player_type == 'Pitcher'], 'war12_per_150ip', f'PIT war{WINDOW}/150ip', 'qualified_pit')

    p2 = p[p.mover.notna()].copy()
    p2['mover'] = p2['mover'].astype(int)
    block(p2[p2.window_censored == 0], f'UNCENSORED (draft <= {max_season - WINDOW}, full {WINDOW}yr)')
    block(p2[p2.window_censored == 1], 'CENSORED (window still open) -- interpret with care')

    pd.DataFrame(rows).to_csv(OUT_CMP, index=False)
    print(f"\nSaved {OUT_CMP}")
    print("\nThe headline is the UNCENSORED block. war12/yr blends quality and")
    print("durability; seasons_in_window is a clean 0-12 longevity score on an")
    print("identical clock for every player.")
    print("Done.")


if __name__ == '__main__':
    main()
