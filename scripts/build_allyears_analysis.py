"""
Build the FULL-HISTORY analysis dataset (every draft year in the register).

The main pipeline builds v3_analysis.csv for the configured cohort window.
This builds the same thing for every year present, writing a separate file so
the main pipeline's outputs are untouched.

Design principle: include everything, discard nothing, and FLAG the things
that make eras non-comparable so they can be examined rather than assumed:

  phase          draft phase as recorded (pre-1987 had January and secondary
                 phases drawing from a different population)
  is_june_reg    convenience flag for the June regular phase
  era            5-year bucket, for grouped reporting
  max_round_yr   the deepest round used in that draft year, so round-depth
                 changes are visible in the data
  censored       1 for classes too recent to have had a fair shot at MLB
                 (default: drafted within CENSOR_YEARS of today). These are
                 kept in the file but should be excluded from reach-rate
                 models -- a 2024 draftee has not failed to reach MLB, he
                 simply has not had time.

Output:
    v3_analysis_allyears.csv

Run:
    python scripts/build_allyears_analysis.py
    python scripts/build_allyears_analysis.py --censor-years 6
"""

import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from geo_tables import apply_typo_fix

REGISTER = 'data/tbc_draft_register.csv'
BONUS = 'data/tbc_signing_bonus.csv'
CACHE_FILE = 'geocode_cache_v3.json'
OUT = 'v3_analysis_allyears.csv'

MOVER_CUTOFF_MI = 5.0
DEFAULT_CENSOR_YEARS = 6      # classes newer than this are flagged censored

WARM_STATES = {'FL', 'TX', 'CA', 'AZ', 'GA', 'NC', 'SC', 'AL', 'MS', 'LA',
               'NV', 'HI', 'PR'}
DOMESTIC = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
            'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
            'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
            'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
            'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}


def arg(flag, default):
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def parse_place(p):
    if pd.isna(p) or ',' not in str(p):
        return None, None
    a, b = str(p).rsplit(',', 1)
    return a.strip(), b.strip()


def parse_school(s):
    if pd.isna(s):
        return None, None, None
    m = re.search(r'\(([^()]+)\)\s*$', str(s))
    if not m:
        return str(s).strip(), None, None
    name = str(s)[:m.start()].strip()
    inside = m.group(1).strip()
    if ',' in inside:
        a, b = inside.rsplit(',', 1)
        return name, a.strip(), b.strip()
    return name, inside, None


def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return np.nan
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)))


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def main():
    censor_years = arg('--censor-years', DEFAULT_CENSOR_YEARS)
    this_year = datetime.date.today().year

    if not os.path.exists(REGISTER):
        print(f"ERROR: {REGISTER} not found. Run from the project root.")
        sys.exit(1)
    df = pd.read_csv(REGISTER, low_memory=False)
    print(f"Register rows: {len(df):,}  years {int(df['year'].min())}-{int(df['year'].max())}")

    # --- locations -------------------------------------------------------
    bp = df['place'].apply(lambda p: pd.Series(parse_place(p)))
    df['birth_city'], df['birth_state'] = bp[0], bp[1]
    sp = df['school'].apply(lambda s: pd.Series(parse_school(s)))
    df['hs_name'], df['hs_city'], df['hs_state'] = sp[0], sp[1], sp[2]

    # signing-bonus hsplace fills in college draftees where available
    reg_pid = pick(df, 'PlayerID', 'playerid')
    if os.path.exists(BONUS) and reg_pid:
        sb = pd.read_csv(BONUS, low_memory=False)
        sb_pid = pick(sb, 'playerid', 'PlayerID')
        if sb_pid and 'hsplace' in sb.columns:
            t = sb[[sb_pid, 'hsplace']].dropna()
            m = dict(zip(t[sb_pid].astype(str).str.strip(),
                         t['hsplace'].astype(str).str.strip()))
            filled = df[reg_pid].astype(str).str.strip().map(m)
            fp = filled.apply(lambda p: pd.Series(parse_place(p)))
            df['hs_city'] = df['hs_city'].fillna(fp[0])
            df['hs_state'] = df['hs_state'].fillna(fp[1])
            print(f"Signing-bonus hsplace filled {fp[0].notna().sum():,} rows")

    # apply the shared typo corrections
    for ccol, scol in (('birth_city', 'birth_state'), ('hs_city', 'hs_state')):
        fixed = [apply_typo_fix(c, s) for c, s in zip(df[ccol], df[scol])]
        df[ccol] = [f[0] for f in fixed]
        df[scol] = [f[1] for f in fixed]

    # --- coordinates -----------------------------------------------------
    if not os.path.exists(CACHE_FILE):
        print(f"ERROR: {CACHE_FILE} not found. Run geocode_all_years.py first.")
        sys.exit(1)
    with open(CACHE_FILE, encoding='utf-8') as f:
        cache = json.load(f)
    coords = {k: (v['lat'], v['lon']) for k, v in cache.items()
              if v.get('lat') is not None and v.get('lon') is not None}
    print(f"Geocode cache: {len(coords):,} resolved places")

    def look(c, s):
        if pd.isna(c) or pd.isna(s):
            return (np.nan, np.nan)
        return coords.get(f"{str(c).strip()}|{str(s).strip()}", (np.nan, np.nan))

    bl = [look(c, s) for c, s in zip(df['birth_city'], df['birth_state'])]
    df['birth_lat'] = [x[0] for x in bl]
    df['birth_lon'] = [x[1] for x in bl]
    hl = [look(c, s) for c, s in zip(df['hs_city'], df['hs_state'])]
    df['hs_lat'] = [x[0] for x in hl]
    df['hs_lon'] = [x[1] for x in hl]

    df['distance_miles'] = [haversine(a, b, c, d) for a, b, c, d in
                            zip(df['birth_lat'], df['birth_lon'],
                                df['hs_lat'], df['hs_lon'])]
    df['log_distance'] = np.log1p(df['distance_miles'])
    df['mover'] = np.where(df['distance_miles'].isna(), np.nan,
                           (df['distance_miles'] > MOVER_CUTOFF_MI).astype(float))

    # --- outcome + classification ---------------------------------------
    df['reached_mlb'] = (df['highLevel'] == 'MLB').astype(int)
    df['is_hs_draftee'] = ((df.get('playerClass') == 'HS') |
                           (df.get('schoolDivision') == 'HS')).astype(int)
    df['is_college_draftee'] = 1 - df['is_hs_draftee']
    if 'signed' in df.columns:
        df['signed_flag'] = (df['signed'].astype(str).str.strip() == 'Y').astype(int)

    # --- era / phase / depth flags (reported, NOT filtered) --------------
    phase_col = pick(df, 'phase', 'Phase')
    if phase_col:
        df['phase'] = df[phase_col].astype(str).str.strip()
        df['is_june_reg'] = df['phase'].str.lower().str.startswith('june-reg').astype(int)
    else:
        df['phase'] = 'unknown'
        df['is_june_reg'] = 1

    df['era'] = (df['year'] // 5 * 5).astype(int)
    df['max_round_yr'] = df.groupby('year')['draftRound'].transform('max')
    df['censored'] = (df['year'] > this_year - censor_years).astype(int)

    # warm-state + foreign flags
    df['birth_warm_state'] = df['birth_state'].apply(
        lambda s: int(str(s).strip().upper() in WARM_STATES) if pd.notna(s) else np.nan)
    df['hs_warm_state'] = df['hs_state'].apply(
        lambda s: int(str(s).strip().upper() in WARM_STATES) if pd.notna(s) else np.nan)
    df['is_foreign_born'] = df['birth_state'].apply(
        lambda s: (0 if str(s).strip().upper() in DOMESTIC else 1)
        if pd.notna(s) else np.nan)

    df.to_csv(OUT, index=False, encoding='utf-8-sig')

    # --- report ----------------------------------------------------------
    print(f"\n{'=' * 84}")
    print("COVERAGE BY ERA (nothing filtered; flags let you slice later)")
    print("=" * 84)
    print(f"{'Era':<10} {'Rows':>8} {'HS':>7} {'HS w/dist':>10} {'cov%':>7} "
          f"{'MaxRnd':>7} {'JuneReg%':>9} {'Censored':>9}")
    print("-" * 84)
    for era, g in df.groupby('era'):
        hs = g[g['is_hs_draftee'] == 1]
        wd = int(hs['distance_miles'].notna().sum())
        print(f"{int(era)}-{int(era)+4:<5} {len(g):>8,} {len(hs):>7,} {wd:>10,} "
              f"{(wd/len(hs)*100 if len(hs) else 0):>6.1f}% "
              f"{int(g['max_round_yr'].max()):>7} "
              f"{g['is_june_reg'].mean()*100:>8.1f}% "
              f"{'yes' if g['censored'].max() else '':>9}")

    hs_all = df[df['is_hs_draftee'] == 1]
    usable = hs_all[hs_all['distance_miles'].notna()]
    print(f"\nTotal HS draftees: {len(hs_all):,}")
    print(f"  with distance:   {len(usable):,} ({len(usable)/len(hs_all)*100:.1f}%)")
    print(f"  uncensored:      {int((usable['censored'] == 0).sum()):,}")
    print(f"  June-Regular:    {int(usable['is_june_reg'].sum()):,}")
    print(f"\nSaved: {OUT}")
    print(f"\nNote: rows drafted after {this_year - censor_years} are flagged "
          f"censored=1.\n      They are kept in the file but should be excluded "
          f"from reach-rate models.\n")


if __name__ == '__main__':
    main()
