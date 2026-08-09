"""
COVERAGE-GAP analysis -- where do draftees reach MLB LESS than their draft slot
predicted, and is that gap better indexed by BIRTH area or HS area?

Actionable framing: a front office cannot change where talent is born, so "which
states produce more MLB players" is not useful (that's talent density). The
actionable quantity is the RESIDUAL: actual reached_mlb minus the reach
probability predicted by draft round. A negative area-mean residual = "draftees
from here bust more than their draft position said they would" = a coverage or
development gap a club can attack.

Expected reach from round: logit(reached_mlb ~ round) fit on the whole cohort,
predicted per player. residual = reached - p_expected. Averaged by area.

Run head-to-head:
  - BIRTH geography vs HS geography (which better predicts the gap)
  - STATE grain and REGION grain, analyzed separately
  - split HITTER vs PITCHER (does the bat gap cluster differently -- the
    actionable "point hitting-eval resources here" question)

Thin-cell guard: only areas with >= MIN_N draftees are ranked.

Run:  python scripts/v3_coverage_gap.py
Outputs: v3_coverage_gap_state.csv, v3_coverage_gap_region.csv
"""

import os
import sys
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

MIN_N_STATE = 40
MIN_N_REGION = 40
PITCHER_POS = {'P', 'RHP', 'LHP', 'PITCHER', 'SP', 'RP'}

# Census region by USPS state abbrev (DC folded into South; territories -> None)
STATE_REGION = {
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
    'PA': 'Northeast',
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
    'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
    'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South', 'NC': 'South',
    'SC': 'South', 'VA': 'South', 'DC': 'South', 'WV': 'South', 'AL': 'South',
    'KY': 'South', 'MS': 'South', 'TN': 'South', 'AR': 'South', 'LA': 'South',
    'OK': 'South', 'TX': 'South',
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West', 'NV': 'West',
    'NM': 'West', 'UT': 'West', 'WY': 'West', 'AK': 'West', 'CA': 'West',
    'HI': 'West', 'OR': 'West', 'WA': 'West',
}


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
    return df


def add_type(df):
    poscol = next((c for c in df.columns if c.lower() in
                   ('draftposit', 'pos', 'position', 'draft_pos')), None)
    if poscol:
        df['dtype'] = df[poscol].apply(
            lambda p: 'Pitcher' if is_pitcher_pos(p) is True
            else ('Hitter' if is_pitcher_pos(p) is False else None))
    else:
        df['dtype'] = None
    return df, poscol


def expected_reach(df):
    """p_expected from logit(reached ~ round); residual = reached - p."""
    rcol = 'draftRound' if 'draftRound' in df.columns else 'round'
    if HAVE_SM:
        m = smf.logit(f'reached_mlb ~ {rcol}', data=df).fit(disp=False)
        df['p_exp'] = m.predict(df)
    else:
        # fallback: empirical reach rate by round
        rate = df.groupby(rcol)['reached_mlb'].transform('mean')
        df['p_exp'] = rate
    df['resid'] = df['reached_mlb'] - df['p_exp']
    return df


def summarize(df, area_col, min_n, split_type):
    """Mean reach residual by area, optionally split by type. Returns tidy df."""
    rows = []
    groups = [('All', df)]
    if split_type and df['dtype'].notna().any():
        groups += [('Hitter', df[df['dtype'] == 'Hitter']),
                   ('Pitcher', df[df['dtype'] == 'Pitcher'])]
    for tlabel, g in groups:
        agg = g.groupby(area_col).agg(
            n=('reached_mlb', 'size'),
            reach=('reached_mlb', 'mean'),
            resid=('resid', 'mean')).reset_index()
        agg = agg[agg['n'] >= min_n]
        agg['type'] = tlabel
        rows.append(agg.rename(columns={area_col: 'area'}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    df = load()
    df, poscol = add_type(df)
    print(f"Rows {COHORT_LABEL}: {len(df):,}   "
          f"draft-type coverage: {df['dtype'].notna().mean()*100:.0f}%")
    df = expected_reach(df)

    # region columns from state
    for who in ['birth', 'school']:
        sc = f'{who}_state'
        if sc in df.columns:
            df[f'{who}_region'] = df[sc].str.upper().str.strip().map(STATE_REGION)

    geos = [('birth', 'birth_state', 'birth_region'),
            ('hs', 'school_state', 'school_region')]

    # ---- STATE grain ----
    print("\n" + "="*76)
    print("STATE grain -- mean reach residual (actual minus round-expected)")
    print("negative = reaches MLB LESS than draft slot predicted (coverage gap)")
    print("="*76)
    state_out = []
    for gkey, scol, rcol in geos:
        if scol not in df.columns:
            continue
        d = df[df[scol].notna()].copy()
        d[scol] = d[scol].str.upper().str.strip()
        s = summarize(d, scol, MIN_N_STATE, split_type=True)
        s['geo'] = gkey
        state_out.append(s)
        allrows = s[s['type'] == 'All'].sort_values('resid')
        print(f"\n[{gkey.upper()} STATE]  most-negative residual (>= {MIN_N_STATE} draftees):")
        print(f"  {'state':<7}{'n':>6}{'reach':>7}{'resid':>8}")
        for _, r in allrows.head(8).iterrows():
            print(f"  {r['area']:<7}{int(r['n']):>6}{r['reach']:>7.3f}{r['resid']:>+8.3f}")
    if state_out:
        pd.concat(state_out, ignore_index=True).to_csv('v3_coverage_gap_state.csv', index=False)

    # ---- REGION grain ----
    print("\n" + "="*76)
    print("REGION grain -- mean reach residual, by type (Hitter vs Pitcher)")
    print("="*76)
    region_out = []
    for gkey, scol, rcol in geos:
        if rcol not in df.columns:
            continue
        d = df[df[rcol].notna()].copy()
        s = summarize(d, rcol, MIN_N_REGION, split_type=True)
        s['geo'] = gkey
        region_out.append(s)
        print(f"\n[{gkey.upper()} REGION]")
        piv = s.pivot_table(index='area', columns='type', values='resid')
        ncnt = s[s['type'] == 'All'].set_index('area')['n']
        print(f"  {'region':<11}{'n':>6}{'All':>8}{'Hitter':>8}{'Pitcher':>8}")
        for area in ['Northeast', 'Midwest', 'South', 'West']:
            if area in piv.index:
                row = piv.loc[area]
                n = int(ncnt.get(area, 0))
                a = row.get('All', np.nan); h = row.get('Hitter', np.nan); p = row.get('Pitcher', np.nan)
                print(f"  {area:<11}{n:>6}{a:>+8.3f}{h:>+8.3f}{p:>+8.3f}")
    if region_out:
        pd.concat(region_out, ignore_index=True).to_csv('v3_coverage_gap_region.csv', index=False)

    print("\n" + "="*76)
    print("READ:")
    print("- Compare BIRTH vs HS: whichever geography shows the sharper residual")
    print("  spread is the better index of the coverage gap. Visibility mechanism")
    print("  predicts HS area carries more signal (scouts go where games are).")
    print("- Hitter vs Pitcher columns: a more-negative HITTER residual in an area")
    print("  = under-projected bats there = point hitting-eval resources at it.")
    print("- Residuals are vs ROUND only; they are a screen, not a causal claim.")
    print("\nSaved: v3_coverage_gap_state.csv, v3_coverage_gap_region.csv")


if __name__ == '__main__':
    main()
