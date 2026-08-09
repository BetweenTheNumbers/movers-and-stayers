"""
Actual player records -- a gut-check on the data and a source of talk anecdotes.

TABLE 1: Top 25 MOVERS by total career WAR
  movers = birth city != HS city (distance > SAME_CITY_MAX). Shows mover
  distance, round, WAR, and the per-rate (WAR/600PA or /150IP) for context.

TABLE 2: Biggest BUSTS among top-100-overall picks
  high pick = overall <= 100 (comp/supplemental picks fall in naturally by
  their real overall number). Ranked by lowest career WAR (floor at reached or
  all, see FLAG). Mover/stayer flag + distance shown so you can eyeball whether
  busts skew mover or stayer.

Everything keyed off real columns, discovered case-insensitively; the script
prints which columns it used so nothing is silently wrong.

Run:  python scripts/v3_player_records.py
Outputs: v3_top_movers_war.csv, v3_top100_busts.csv
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
PA_FLOOR = 145
IP_FLOOR = 50
TOP_N = 50
HIGH_PICK_MAX = 100     # top-100 overall = "high pick"


def find(df, *names):
    """First column whose lower-case name matches any candidate."""
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


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
    return df


def main():
    df = load()

    # resolve columns
    fn = find(df, 'firstName', 'first_name', 'first')
    ln = find(df, 'lastName', 'last_name', 'last')
    dist = find(df, 'dist', 'distance_mi', 'distance_miles', 'distance')
    war = find(df, 'career_war', 'war')
    hwar = find(df, 'hit_war')
    pwar = find(df, 'pit_war')
    pa = find(df, 'hit_pa', 'career_pa', 'pa')
    ip = find(df, 'pit_ip', 'career_ip', 'ip')
    rnd = find(df, 'draftRound', 'round')
    overall = find(df, 'overall', 'overallPick', 'pick', 'overall_pick')
    yr = find(df, 'year')
    bcity = find(df, 'birth_city')
    hcity = find(df, 'school_city')
    bstate = find(df, 'birth_state')
    hstate = find(df, 'school_state')

    print("Columns used:")
    print(f"  name={fn}/{ln}  dist={dist}  war={war}  hit_war={hwar} pit_war={pwar}")
    print(f"  pa={pa} ip={ip}  round={rnd}  overall={overall}  year={yr}")
    if war is None or dist is None:
        print("\nERROR: need at least a WAR column and a distance column.")
        sys.exit(1)

    df['_name'] = (df[fn].fillna('').astype(str) + ' ' +
                   df[ln].fillna('').astype(str)).str.strip() if fn and ln else '(no name)'
    df['_mover'] = np.where(df[dist] > SAME_CITY_MAX, 'MOVER', 'stayer')
    df['_from'] = ''
    if bcity and bstate:
        df['_from'] = df[bcity].fillna('') + ', ' + df[bstate].fillna('')
    df['_hs'] = ''
    if hcity and hstate:
        df['_hs'] = df[hcity].fillna('') + ', ' + df[hstate].fillna('')

    # per-rate for context
    df['_rate'] = np.nan
    if hwar and pa:
        m = df[pa].fillna(0) >= PA_FLOOR
        df.loc[m, '_rate'] = df.loc[m, hwar] / df.loc[m, pa] * 600
    if pwar and ip:
        m = df[ip].fillna(0) >= IP_FLOOR
        df.loc[m, '_rate'] = df.loc[m, pwar] / df.loc[m, ip] * 150

    # ---- TABLE 1: top movers by career WAR ----
    movers = df[(df['_mover'] == 'MOVER') & df[war].notna()].copy()
    top = movers.nlargest(TOP_N, war)
    print("\n" + "="*94)
    print(f"TOP {TOP_N} MOVERS BY CAREER WAR   ({COHORT_LABEL})")
    print("="*94)
    print(f"{'#':>2} {'player':<22}{'yr':>5}{'rnd':>4}{'dist_mi':>9}"
          f"{'WAR':>7}{'rate':>7}  {'born -> HS'}")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        rate = f"{r['_rate']:.2f}" if pd.notna(r['_rate']) else "  -"
        route = f"{r['_from']} -> {r['_hs']}" if r['_from'] else ""
        print(f"{i:>2} {r['_name'][:22]:<22}{int(r[yr]):>5}"
              f"{int(r[rnd]) if pd.notna(r[rnd]) else 0:>4}{r[dist]:>9.0f}"
              f"{r[war]:>7.1f}{rate:>7}  {route}")
    cols1 = ['_name', yr, rnd, dist, war, '_rate', '_from', '_hs']
    top[cols1].to_csv('v3_top_movers_war.csv', index=False)

    # ---- TABLE 2: busts among top-100 overall ----
    if overall is None:
        print("\n(NOTE: no 'overall' pick column found -- using rounds 1-3 as the")
        print(" high-pick proxy for the bust table.)")
        highpick = df[df[rnd] <= 3].copy()
        pick_desc = "rounds 1-3"
    else:
        df[overall] = pd.to_numeric(df[overall], errors='coerce')
        highpick = df[df[overall] <= HIGH_PICK_MAX].copy()
        pick_desc = f"top {HIGH_PICK_MAX} overall"

    # bust = high pick with the lowest career WAR (negatives AND zeros
    # included), sorted ascending. A player who REACHED MLB but has a blank WAR
    # is a genuine ~0 contributor, so fill him to 0.0 rather than drop him.
    # Only players with no WAR AND no MLB appearance (never produced) are excluded.
    reached = find(df, 'reached_mlb')
    hp = highpick.copy()
    if reached:
        reached_blank = hp[war].isna() & (hp[reached] == 1)
        hp.loc[reached_blank, war] = 0.0
    have_war = hp[hp[war].notna()].copy()
    busts = have_war.sort_values(war).head(TOP_N)
    n_nowar = int(hp[war].isna().sum())
    print("\n" + "="*94)
    print(f"BIGGEST BUSTS AMONG {pick_desc.upper()} PICKS   (lowest career WAR, ascending)")
    print(f"(zeros and negatives included; {n_nowar} {pick_desc} picks never reached "
          f"MLB / no WAR -- excluded)")
    print("="*94)
    print(f"{'#':>2} {'player':<22}{'yr':>5}{'ovr':>5}{'rnd':>4}"
          f"{'WAR':>7}  {'M/S':<6}{'dist_mi':>8}  {'born -> HS'}")
    for i, (_, r) in enumerate(busts.iterrows(), 1):
        warv = f"{r[war]:.1f}" if pd.notna(r[war]) else "none"
        ov = int(r[overall]) if overall and pd.notna(r[overall]) else 0
        route = f"{r['_from']} -> {r['_hs']}" if r['_from'] else ""
        print(f"{i:>2} {r['_name'][:22]:<22}{int(r[yr]):>5}{ov:>5}"
              f"{int(r[rnd]) if pd.notna(r[rnd]) else 0:>4}{warv:>7}  "
              f"{r['_mover']:<6}{r[dist]:>8.0f}  {route}")
    cols2 = ['_name', yr, overall, rnd, war, '_mover', dist, '_from', '_hs']
    cols2 = [c for c in cols2 if c]
    busts[cols2].to_csv('v3_top100_busts.csv', index=False)

    # quick tally: what share of the busts were movers?
    share = (busts['_mover'] == 'MOVER').mean()
    base = (highpick['_mover'] == 'MOVER').mean()
    print(f"\nMover share among these busts: {share*100:.0f}%   "
          f"(vs {base*100:.0f}% of all {pick_desc} picks)")
    print("If bust-mover-share is BELOW the baseline, movers bust less among high")
    print("picks -- consistent with movers being better-evaluated (more visible).")

    print("\nSaved: v3_top_movers_war.csv, v3_top100_busts.csv")


if __name__ == '__main__':
    main()
