"""
Mover tiers, SPLIT BY PLAYER TYPE (hitter vs pitcher).

The pooled tier analysis (v3_mover_tiers.py) showed arrival rises with distance
while position-player per-PA value falls at the far tier, but pitcher per-IP
value is flat. This script tests that split properly:

  ARRIVAL  -- split all draftees by DRAFT-TIME position (register 'Pos' field:
              P/RHP/LHP = pitcher, else hitter), so we're not conditioning on
              reaching MLB. Separate logit per type.
  VALUE    -- among reachers, split by FanGraphs career profile (more IP-heavy
              => pitcher). WAR/600 PA for hitters, WAR/150 IP for pitchers.
              (FanGraphs house rate conventions.)

Tiers: Stayer (<=1mi) / Short (<=50) / Medium (50-500) / Long (>500).
Round-controlled throughout.

Outputs:
  v3_tier_bytype_arrival.csv
  v3_tier_bytype_value.csv

Run:  python scripts/v3_mover_tiers_bytype.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd

try:
    from config import START_YEAR, END_YEAR, COHORT_LABEL
except Exception:
    START_YEAR, END_YEAR, COHORT_LABEL = 1996, 2019, "1996-2019"

try:
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

PA_QUAL = 145
IP_QUAL = 50
SAME_CITY_MAX = 1.0
EDGES = [50, 500]
ORDER = ['0 Stayer', '1 Short', '2 Medium', '3 Long']

PITCHER_POS = {'P', 'RHP', 'LHP', 'PITCHER', 'SP', 'RP', 'LHP/RHP'}


def is_pitcher_pos(p):
    """True if a draft-position string denotes a pitcher, across common formats."""
    if p is None:
        return None
    s = str(p).upper().strip()
    if s in ('', 'NAN', 'NONE'):
        return None
    if s in PITCHER_POS:
        return True
    # 'P', 'RHP', 'LHP', 'P-...', 'RHP/1B' (two-way listed pitcher-first) etc.
    if s == 'P' or s.startswith('RHP') or s.startswith('LHP') or s.startswith('P/') \
       or s.startswith('P-') or s.endswith('HP'):
        return True
    return False


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
    dcol = next((c for c in ['distance_mi', 'distance_miles', 'dist_mi', 'distance']
                 if c in df.columns), None)
    df = df[df[dcol].notna()].rename(columns={dcol: 'dist'})
    print(f"Source: {src}   HS draftees w/ distance {COHORT_LABEL}: {len(df):,}")
    return df


def tier(d):
    if d <= SAME_CITY_MAX:
        return '0 Stayer'
    names = ['1 Short', '2 Medium', '3 Long']
    for i, e in enumerate(EDGES):
        if d <= e:
            return names[i]
    return names[len(EDGES)]


def draft_type(row, poscol):
    """Hitter/Pitcher from draft-time register position if present."""
    if poscol and poscol in row and pd.notna(row[poscol]):
        v = is_pitcher_pos(row[poscol])
        if v is True:
            return 'Pitcher'
        if v is False:
            return 'Hitter'
    return None


def find_poscol(df):
    """Case-insensitive hunt for a draft position column; else pull from register."""
    for c in df.columns:
        if c.lower() in ('pos','position','draft_pos','primarypos','pos_draft','draftposit','draftposition'):
            return df, c
    # not in the analysis file -> try to join it from the register
    reg_path = 'data/tbc_draft_register.csv'
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path, low_memory=False)
        rpos = next((c for c in reg.columns if c.lower() in
                     ('pos', 'position')), None)
        # find a shared key
        key = next((k for k in ['mlbid', 'register_id', 'playerid', 'bbref_id']
                    if k in df.columns and k in reg.columns), None)
        if rpos and key:
            df = df.merge(reg[[key, rpos]].drop_duplicates(key),
                          on=key, how='left')
            print(f"  joined '{rpos}' from register on '{key}'")
            return df, rpos
    return df, None


def main():
    df = load()
    df['tier'] = df['dist'].apply(tier)

    # draft-time type for the arrival split
    df, poscol = find_poscol(df)
    if poscol is not None:
        vals = df[poscol].astype(str).str.upper().str.strip().value_counts().head(25)
        print(f"\nDraft-position column: '{poscol}'. Top values seen:")
        print("  " + ", ".join(f"{k}:{v}" for k, v in vals.items()))
    if poscol is None:
        print("\nWARNING: no draft-position column found in analysis file OR register.")
        print("Arrival-by-type CANNOT be trusted (career-type fallback conditions on")
        print("reaching MLB). Reporting VALUE split only; skipping arrival ORs.")
    df['dtype'] = df.apply(lambda r: draft_type(r, poscol), axis=1)
    have_pos = df['dtype'].notna().mean()
    if have_pos < 0.5:
        print(f"\nNOTE: draft position present for only {have_pos*100:.0f}% of rows.")
        print("Columns in analysis file (to help locate a position field):")
        print("  " + ", ".join(df.columns[:40]))
        print("SKIPPING arrival-by-type (career fallback is invalid for arrival).")
        print("The VALUE-by-type split below is still valid.\n")
        arrival_ok = False
    else:
        arrival_ok = True

    # career type (for value, and as fallback for arrival)
    hw = df['hit_war'] if 'hit_war' in df.columns else np.nan
    ip = df['pit_ip'] if 'pit_ip' in df.columns else 0
    pa = df['hit_pa'] if 'hit_pa' in df.columns else 0
    df['career_type'] = np.where(pd.Series(ip).fillna(0)*4.3 >=
                                 pd.Series(pa).fillna(0), 'Pitcher', 'Hitter')
    df['dtype'] = df['dtype'].fillna(df['career_type'])

    print(f"\nDraft-time type available: {have_pos*100:.0f}%   "
          f"Hitters {int((df.dtype=='Hitter').sum()):,} / "
          f"Pitchers {int((df.dtype=='Pitcher').sum()):,}")

    # ---- ARRIVAL by type ----
    if not arrival_ok:
        print("(arrival-by-type skipped -- see note above)")
        arows = []
    else:
        print("\n" + "="*78)
        print("ARRIVAL by tier, split by player type")
        print("="*78)
        arows = []
        for typ in ['Hitter', 'Pitcher']:
            sub = df[df['dtype'] == typ]
            print(f"\n{typ}s (n={len(sub):,}):")
            print(f"  {'tier':<10} {'N':>6} {'MLB%':>7}")
            for t in ORDER:
                ts = sub[sub['tier'] == t]
                if len(ts) < 20:
                    continue
                print(f"  {t:<10} {len(ts):>6} {ts['reached_mlb'].mean()*100:>6.1f}%")
                arows.append({'type': typ, 'tier': t, 'n': len(ts),
                              'mlb_rate': round(ts['reached_mlb'].mean(), 4)})
            if HAVE_SM and len(sub) > 100:
                sub2 = sub.copy()
                sub2['tier'] = pd.Categorical(sub2['tier'], categories=ORDER)
                try:
                    m = smf.logit('reached_mlb ~ C(tier, Treatment("0 Stayer")) + draftRound',
                                  data=sub2).fit(disp=False)
                    print(f"  round-controlled ORs vs Stayer:")
                    for nm in m.params.index:
                        if 'tier' in nm:
                            lbl = nm.split('T.')[-1].rstrip(']')
                            print(f"    {lbl:<10} OR={np.exp(m.params[nm]):.2f}  "
                                  f"p={m.pvalues[nm]:.4f}")
                except Exception as e:
                    print(f"  model failed: {e}")
    pd.DataFrame(arows).to_csv('v3_tier_bytype_arrival.csv', index=False)

    # ---- VALUE by type (reachers) ----
    print("\n" + "="*78)
    print("VALUE by tier, split by player type (MLB reachers)")
    print("="*78)
    r = df[df['reached_mlb'] == 1].copy()
    if 'hit_pa' in r.columns:
        r['career_pa'] = r['hit_pa']
    if 'pit_ip' in r.columns:
        r['career_ip'] = r['pit_ip']
    r['_hwar'] = r['hit_war'] if 'hit_war' in r.columns else r['career_war']
    r['_pwar'] = r['pit_war'] if 'pit_war' in r.columns else r['career_war']

    vrows = []
    # hitters: WAR/600 PA
    print("\nHitters -- WAR per 600 PA (qualified >=145 PA):")
    print(f"  {'tier':<10} {'nQual':>6} {'W/600PA':>8}")
    rh = r[r['career_type'] == 'Hitter'].copy()
    rh['w600'] = np.where(rh['career_pa'] >= PA_QUAL,
                          rh['_hwar']/rh['career_pa']*600, np.nan)
    for t in ORDER:
        ts = rh[(rh['tier'] == t) & rh['w600'].notna()]
        if len(ts) < 10:
            continue
        print(f"  {t:<10} {len(ts):>6} {ts['w600'].mean():>8.2f}")
        vrows.append({'type': 'Hitter', 'tier': t, 'n_qual': len(ts),
                      'war_rate': round(ts['w600'].mean(), 3), 'per': '600PA'})
    # pitchers: WAR/150 IP
    print("\nPitchers -- WAR per 150 IP (qualified >=50 IP):")
    print(f"  {'tier':<10} {'nQual':>6} {'W/150IP':>8}")
    rp = r[r['career_type'] == 'Pitcher'].copy()
    rp['w150'] = np.where(rp['career_ip'] >= IP_QUAL,
                          rp['_pwar']/rp['career_ip']*150, np.nan)
    for t in ORDER:
        ts = rp[(rp['tier'] == t) & rp['w150'].notna()]
        if len(ts) < 10:
            continue
        print(f"  {t:<10} {len(ts):>6} {ts['w150'].mean():>8.2f}")
        vrows.append({'type': 'Pitcher', 'tier': t, 'n_qual': len(ts),
                      'war_rate': round(ts['w150'].mean(), 3), 'per': '150IP'})
    pd.DataFrame(vrows).to_csv('v3_tier_bytype_value.csv', index=False)

    print("\n" + "="*78)
    print("READ: if arrival rises with distance for BOTH types, but per-unit value")
    print("falls only for HITTERS, the visibility effect is a position-player story")
    print("(bat value is harder to project than arm talent, so 'seen-not-good'")
    print("has more room to operate on hitters).")
    print("\nSaved: v3_tier_bytype_arrival.csv, v3_tier_bytype_value.csv")
    print("Done.")


if __name__ == '__main__':
    main()
