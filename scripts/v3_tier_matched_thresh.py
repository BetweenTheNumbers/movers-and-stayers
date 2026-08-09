"""
STRESS TEST: does the hitter-declines / pitcher-flat value split survive when
hitters and pitchers face a PROPORTIONALLY EQUAL "meaningful MLB sample" bar?

The main tier script qualifies hitters at 145 PA and pitchers at 50 IP. Those
are NOT proportionally equal: at ~4.2 PA per inning faced, 145 PA ~= 35 IP, so
50 IP is a stricter entry bar for pitchers. If the hitter value decline is an
artifact of that mismatch, matched bars will erase it. If it is real, it holds.

We test three matched pairings so the result is not sensitive to the anchor:
  A. anchor on the hitter bar:   145 PA  <-> 35 IP
  B. anchor on the pitcher bar:  210 PA  <-> 50 IP
  C. a middle option:            175 PA  <-> 42 IP
Conversion: 1 IP ~= 4.2 PA faced.

For each pairing we recompute WAR/600 PA (hitters) and WAR/150 IP (pitchers) by
tier, with qualifier counts, so thin cells are visible. The rate DENOMINATORS
(600 / 150) never change -- only the qualification FLOOR that decides who is in
the comparison.

Run:  python scripts/v3_tier_matched_thresh.py
Output: v3_tier_matched_thresh.csv
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
PA_PER_IP = 4.2

# (label, PA floor, IP floor) -- proportionally matched pairings
PAIRINGS = [
    ('A hitter-anchored', 145, 145/PA_PER_IP),   # 145 PA <-> ~35 IP
    ('B pitcher-anchored', 50*PA_PER_IP, 50),     # ~210 PA <-> 50 IP
    ('C middle',          175, 175/PA_PER_IP),    # 175 PA <-> ~42 IP
]

PITCHER_POS = {'P', 'RHP', 'LHP', 'PITCHER', 'SP', 'RP'}


def is_pitcher_pos(p):
    if p is None:
        return None
    s = str(p).upper().strip()
    if s in ('', 'NAN', 'NONE'):
        return None
    if s in PITCHER_POS:
        return True
    if s == 'P' or s.startswith('RHP') or s.startswith('LHP') \
       or s.startswith('P/') or s.startswith('P-') or s.endswith('HP'):
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
    return df


def tier(d):
    if d <= SAME_CITY_MAX:
        return '0 Stayer'
    names = ['1 Short', '2 Medium', '3 Long']
    for i, e in enumerate(EDGES):
        if d <= e:
            return names[i]
    return names[len(EDGES)]


def main():
    df = load()
    print(f"Source rows {COHORT_LABEL}: {len(df):,}")
    df['tier'] = df['dist'].apply(tier)

    r = df[df['reached_mlb'] == 1].copy()
    # career playing time + component WAR
    r['pa'] = r['hit_pa'] if 'hit_pa' in r.columns else np.nan
    r['ip'] = r['pit_ip'] if 'pit_ip' in r.columns else np.nan
    r['hwar'] = r['hit_war'] if 'hit_war' in r.columns else r['career_war']
    r['pwar'] = r['pit_war'] if 'pit_war' in r.columns else r['career_war']
    # career-profile type (well-defined among reachers)
    r['ctype'] = np.where(r['ip'].fillna(0)*PA_PER_IP >= r['pa'].fillna(0),
                          'Pitcher', 'Hitter')

    rows = []
    for label, pa_floor, ip_floor in PAIRINGS:
        print("\n" + "="*72)
        print(f"PAIRING {label}:  hitters >= {pa_floor:.0f} PA   "
              f"pitchers >= {ip_floor:.0f} IP")
        print("="*72)

        rh = r[r['ctype'] == 'Hitter'].copy()
        rh['w600'] = np.where(rh['pa'] >= pa_floor, rh['hwar']/rh['pa']*600, np.nan)
        rp = r[r['ctype'] == 'Pitcher'].copy()
        rp['w150'] = np.where(rp['ip'] >= ip_floor, rp['pwar']/rp['ip']*150, np.nan)

        print(f"\n  Hitters WAR/600PA          Pitchers WAR/150IP")
        print(f"  {'tier':<9}{'nQ':>5}{'rate':>7}     {'tier':<9}{'nQ':>5}{'rate':>7}")
        for t in ORDER:
            hs = rh[(rh['tier'] == t) & rh['w600'].notna()]
            ps = rp[(rp['tier'] == t) & rp['w150'].notna()]
            hrate = f"{hs['w600'].mean():.2f}" if len(hs) else "  -"
            prate = f"{ps['w150'].mean():.2f}" if len(ps) else "  -"
            print(f"  {t:<9}{len(hs):>5}{hrate:>7}     {t:<9}{len(ps):>5}{prate:>7}")
            rows.append({'pairing': label, 'tier': t,
                         'pa_floor': round(pa_floor), 'ip_floor': round(ip_floor),
                         'hit_nqual': len(hs),
                         'hit_war600': round(hs['w600'].mean(), 3) if len(hs) else None,
                         'pit_nqual': len(ps),
                         'pit_war200': round(ps['w150'].mean(), 3) if len(ps) else None})

        # summarize the split for this pairing
        hh = rh[rh['w600'].notna()]
        st = hh[hh['tier'] == '0 Stayer']['w600'].mean()
        lg = hh[hh['tier'] == '3 Long']['w600'].mean()
        if pd.notna(st) and pd.notna(lg):
            print(f"\n  -> Hitter Stayer {st:.2f} vs Long {lg:.2f}  "
                  f"(ratio {lg/st:.2f})  {'DECLINE HOLDS' if lg < st*0.85 else 'weak/none'}")
        pp = rp[rp['w150'].notna()]
        pst = pp[pp['tier'] == '0 Stayer']['w150'].mean()
        plg = pp[pp['tier'] == '3 Long']['w150'].mean()
        if pd.notna(pst) and pd.notna(plg):
            print(f"  -> Pitcher Stayer {pst:.2f} vs Long {plg:.2f} "
                  f"(ratio {plg/pst:.2f})  "
                  f"{'FLAT' if abs(plg-pst) < pst*0.15 else 'moves'}")

    pd.DataFrame(rows).to_csv('v3_tier_matched_thresh.csv', index=False)
    print("\n" + "="*72)
    print("READ: if hitter Long < ~0.85x stayer across ALL THREE pairings while")
    print("pitchers stay flat, the split is NOT a threshold artifact -- it's real.")
    print("Watch nQ on the Long hitter cell; it's the smallest and drives caution.")
    print("\nSaved: v3_tier_matched_thresh.csv")


if __name__ == '__main__':
    main()
