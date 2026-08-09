"""
Season-level FanGraphs rollup -> per-player RATE metrics, joined to the draft
analysis. Decomposes the career-WAR null into quality-per-unit vs longevity.

Files (correct by CONTENT; names on disk may differ):
  hitting season file  has PA / wOBA / wRC+   (fg-hit-seasons.csv)
  pitching season file has IP / ERA / FIP     (fg-pit-seasons.csv)
Both are already one row per (PlayerId, Season) with traded seasons collapsed
into the "- - -" total row (verified by check_totals.py), so we use rows as-is.

Player typing (by where career playing time sits):
  Hitter   - career PA share dominates
  Pitcher  - career IP share dominates
  Two-Way  - meaningful both (e.g. Ohtani); reported separately AND in combined

Qualification (user rule: rookie graduation ~130 AB / 50 IP):
  PA files give PA, not AB. 130 AB ~= 145 PA (AB + ~12% walks/HBP/sac).
  qualified_hit = career PA >= 145
  qualified_pit = career IP >= 50
  A player qualifies if they clear the bar for their type (either, for two-way).
  Non-qualifiers are KEPT in the file, flagged below_rookie_threshold=1, and
  EXCLUDED from rate comparisons (avoids cup-of-coffee rate garbage).

Rates produced (per qualified player, career aggregated from seasons):
  war_per_season   = total WAR / MLB seasons played
  war_per_150g     = total WAR / total games * 150
  war_per_600pa    = hit WAR / total PA * 600      (hitters)
  war_per_150ip    = pit WAR / total IP * 150      (pitchers)

Joins onto v3_analysis_with_war.csv by MLBAMID so mover/stayer flags are present.

Outputs:
  v3_player_rates.csv                 one row per MLB player with rates + type
  v3_rate_comparison_by_mover.csv     mover vs stayer on each rate, by type

Run:  python scripts/v3_season_rates.py
"""

import os
import sys
import numpy as np
import pandas as pd

HIT_SEASONS = 'fg-hit-seasons.csv'   # content = hitting (PA/wOBA/wRC+)
PIT_SEASONS = 'fg-pit-seasons.csv'   # content = pitching (IP/ERA/FIP)
ANALYSIS = 'v3_analysis_with_war.csv'
OUT_RATES = 'v3_player_rates.csv'
OUT_CMP = 'v3_rate_comparison_by_mover.csv'

PA_QUAL = 145     # ~130 AB
IP_QUAL = 50
TWO_WAY_PA = 145  # to be flagged two-way, need real time on both sides
TWO_WAY_IP = 50


def detect_kind(cols):
    cl = {c.lower() for c in cols}
    if {'era', 'fip', 'ip'} & cl:
        return 'PIT'
    if {'woba', 'wrc+', 'pa', 'obp'} & cl:
        return 'HIT'
    return '?'


def load_seasons(fname, expect):
    if not os.path.exists(fname):
        print(f"ERROR: {fname} not found.")
        sys.exit(1)
    d = pd.read_csv(fname, low_memory=False)
    k = detect_kind(d.columns)
    if k != expect:
        # names may be swapped; trust content, warn
        print(f"  NOTE: {fname} content looks like {k}, expected {expect}; "
              f"using by content.")
    return d


def main():
    # ---- load season files, keyed by content not name --------------------
    a = load_seasons(HIT_SEASONS, 'HIT')
    b = load_seasons(PIT_SEASONS, 'PIT')
    hit = a if detect_kind(a.columns) == 'HIT' else b
    pit = b if detect_kind(b.columns) == 'PIT' else a
    print(f"Hitting seasons: {len(hit):,}   Pitching seasons: {len(pit):,}")

    idc = 'MLBAMID' if 'MLBAMID' in hit.columns else 'PlayerId'
    for d in (hit, pit):
        d[idc] = pd.to_numeric(d[idc], errors='coerce')

    # ---- aggregate seasons -> per player ---------------------------------
    hit_ok = hit.dropna(subset=[idc])
    pit_ok = pit.dropna(subset=[idc])

    hg = hit_ok.groupby(idc).agg(
        hit_war=('WAR', 'sum'),
        hit_pa=('PA', 'sum'),
        hit_g=('G', 'sum'),
        hit_seasons=('Season', 'nunique'),
        first_hit_season=('Season', 'min'),
        last_hit_season=('Season', 'max'),
    ).reset_index()

    pg = pit_ok.groupby(idc).agg(
        pit_war=('WAR', 'sum'),
        pit_ip=('IP', 'sum'),
        pit_g=('G', 'sum'),
        pit_seasons=('Season', 'nunique'),
        first_pit_season=('Season', 'min'),
        last_pit_season=('Season', 'max'),
    ).reset_index()

    p = hg.merge(pg, on=idc, how='outer')
    for c in ['hit_war', 'hit_pa', 'hit_g', 'pit_war', 'pit_ip', 'pit_g']:
        p[c] = p[c].fillna(0)
    p['total_war'] = p['hit_war'] + p['pit_war']
    p['total_g'] = p['hit_g'] + p['pit_g']

    # seasons = union of hit and pit seasons actually played
    p['first_season'] = p[['first_hit_season', 'first_pit_season']].min(axis=1)
    p['last_season'] = p[['last_hit_season', 'last_pit_season']].max(axis=1)
    # a robust season count: max of the two (they overlap for two-way)
    p['mlb_seasons'] = p[['hit_seasons', 'pit_seasons']].max(axis=1).fillna(0)

    # ---- player type -----------------------------------------------------
    def ptype(r):
        h = r['hit_pa'] >= TWO_WAY_PA
        pi = r['pit_ip'] >= TWO_WAY_IP
        if h and pi:
            return 'Two-Way'
        if r['pit_ip'] >= r['hit_pa'] / 3.0 and r['pit_ip'] > 0:
            # pitchers accrue far fewer PA; use IP dominance heuristic
            return 'Pitcher' if r['pit_ip'] >= IP_QUAL or r['pit_ip'] > 0 and r['hit_pa'] < TWO_WAY_PA else 'Hitter'
        return 'Hitter'
    # simpler, more robust: whoever has the qualifying exposure; tie -> bigger WAR side
    def ptype2(r):
        h_ok = r['hit_pa'] >= PA_QUAL
        p_ok = r['pit_ip'] >= IP_QUAL
        if h_ok and p_ok:
            return 'Two-Way'
        if p_ok and not h_ok:
            return 'Pitcher'
        if h_ok and not p_ok:
            return 'Hitter'
        # neither qualifies: classify by whichever exposure is larger (for typing only)
        return 'Pitcher' if r['pit_ip'] * 4.3 >= r['hit_pa'] else 'Hitter'
    p['player_type'] = p.apply(ptype2, axis=1)

    # ---- qualification ---------------------------------------------------
    p['qualified_hit'] = (p['hit_pa'] >= PA_QUAL).astype(int)
    p['qualified_pit'] = (p['pit_ip'] >= IP_QUAL).astype(int)
    p['qualified'] = ((p['qualified_hit'] == 1) | (p['qualified_pit'] == 1)).astype(int)
    p['below_rookie_threshold'] = (p['qualified'] == 0).astype(int)

    # ---- rates (guard denominators) --------------------------------------
    def safe(n, d):
        return np.where((d > 0), n / np.where(d == 0, np.nan, d), np.nan)

    p['war_per_season'] = safe(p['total_war'], p['mlb_seasons'])
    p['war_per_150g'] = safe(p['total_war'], p['total_g']) * 150
    p['war_per_600pa'] = safe(p['hit_war'], p['hit_pa']) * 600
    p['war_per_150ip'] = safe(p['pit_war'], p['pit_ip']) * 150

    # rates only meaningful for qualifiers; null them out otherwise
    for c in ['war_per_season', 'war_per_150g']:
        p.loc[p['qualified'] == 0, c] = np.nan
    p.loc[p['qualified_hit'] == 0, 'war_per_600pa'] = np.nan
    p.loc[p['qualified_pit'] == 0, 'war_per_150ip'] = np.nan

    p = p.rename(columns={idc: 'MLBAMID'})
    p.to_csv(OUT_RATES, index=False)
    print(f"\nSaved {OUT_RATES}: {len(p):,} MLB players")
    print(f"  Hitters: {(p.player_type=='Hitter').sum():,}   "
          f"Pitchers: {(p.player_type=='Pitcher').sum():,}   "
          f"Two-Way: {(p.player_type=='Two-Way').sum():,}")
    print(f"  Qualified (>= {PA_QUAL} PA or {IP_QUAL} IP): {p.qualified.sum():,}   "
          f"Below rookie threshold: {p.below_rookie_threshold.sum():,}")

    # ---- join mover/stayer ----------------------------------------------
    if not os.path.exists(ANALYSIS):
        print(f"\n(no {ANALYSIS}; skipping mover comparison)")
        return
    d = pd.read_csv(ANALYSIS, low_memory=False)
    d = d[(d.get('is_hs_draftee', 0) == 1) & (d['reached_mlb'] == 1)].copy()
    idcol = 'mlbid_clean' if 'mlbid_clean' in d.columns else 'mlbid'
    d[idcol] = pd.to_numeric(d[idcol], errors='coerce')
    # one draft row per player (a player can appear for multiple draft years)
    d = d.sort_values('year').drop_duplicates(subset=[idcol], keep='first')
    m = d.merge(p, left_on=idcol, right_on='MLBAMID', how='inner')
    m = m[m['mover'].notna()].copy()
    m['mover'] = m['mover'].astype(int)
    print(f"\nJoined to {len(m):,} MLB-reaching HS draftees with rates")

    # ---- comparison ------------------------------------------------------
    from scipy import stats
    rows = []
    print(f"\n{'='*84}")
    print("MOVER vs STAYER — RATE METRICS (qualified players only)")
    print("="*84)

    def compare(sub, rate_col, type_label, need_col=None):
        s = sub.copy()
        if need_col:
            s = s[s[need_col] == 1]
        s = s[s[rate_col].notna()]
        mv = s[s['mover'] == 1][rate_col]
        st = s[s['mover'] == 0][rate_col]
        if len(mv) < 10 or len(st) < 10:
            return
        t, pval = stats.ttest_ind(mv, st, equal_var=False)
        print(f"\n{type_label} — {rate_col}")
        print(f"  Movers:  {mv.mean():.3f}  (n={len(mv)})")
        print(f"  Stayers: {st.mean():.3f}  (n={len(st)})")
        print(f"  Diff:    {mv.mean()-st.mean():+.3f}   t-test p={pval:.4f}")
        rows.append({'group': type_label, 'metric': rate_col,
                     'mover_mean': round(mv.mean(), 4), 'mover_n': len(mv),
                     'stayer_mean': round(st.mean(), 4), 'stayer_n': len(st),
                     'diff': round(mv.mean()-st.mean(), 4), 'p_value': pval})

    # combined (all qualified)
    q = m[m['qualified'] == 1]
    compare(q, 'war_per_season', 'ALL (combined)')
    compare(q, 'war_per_150g', 'ALL (combined)')
    # by type
    compare(m[m['player_type'] == 'Hitter'], 'war_per_season', 'Hitters', 'qualified_hit')
    compare(m[m['player_type'] == 'Hitter'], 'war_per_600pa', 'Hitters', 'qualified_hit')
    compare(m[m['player_type'] == 'Pitcher'], 'war_per_season', 'Pitchers', 'qualified_pit')
    compare(m[m['player_type'] == 'Pitcher'], 'war_per_150ip', 'Pitchers', 'qualified_pit')

    # longevity, to decompose the career-total null
    print(f"\n{'='*84}")
    print("LONGEVITY (does career length differ?)")
    print("="*84)
    for lab, col in [('MLB seasons', 'mlb_seasons'), ('Career games', 'total_g')]:
        s = q[q[col].notna()]
        mv, st = s[s.mover == 1][col], s[s.mover == 0][col]
        if len(mv) >= 10 and len(st) >= 10:
            t, pval = stats.ttest_ind(mv, st, equal_var=False)
            print(f"\n{lab}: movers {mv.mean():.2f} vs stayers {st.mean():.2f}  "
                  f"(diff {mv.mean()-st.mean():+.2f}, p={pval:.4f})")
            rows.append({'group': 'longevity', 'metric': col,
                         'mover_mean': round(mv.mean(), 3), 'mover_n': len(mv),
                         'stayer_mean': round(st.mean(), 3), 'stayer_n': len(st),
                         'diff': round(mv.mean()-st.mean(), 3), 'p_value': pval})

    pd.DataFrame(rows).to_csv(OUT_CMP, index=False)
    print(f"\nSaved {OUT_CMP}")
    print("\nReading guide: if rates are equal AND longevity is equal, the")
    print("career-WAR null means truly identical players. If rates differ but")
    print("cancel against longevity, 'identical careers' hides a real difference.")
    print("Done.")


if __name__ == '__main__':
    main()
