"""
Fixed HS-tier analysis for the configured cohort (see scripts/config.py).

The previous C5 model failed because the HS-tier variable had look-ahead bias
that caused complete separation. This script computes HS quality properly
using TWO leakage-free methods and re-runs the regression:

  Method A: Leave-one-out (LOO).
    For each player, count MLB players from his HS *excluding himself*.
    This removes the trivial "this kid made MLB and his HS counts him"
    self-reference but still uses future history of his classmates.

  Method B: Prior-history-only (PRIOR).
    For each player drafted in year Y, count MLB players from his HS who
    were drafted in years < Y. This is the strictest leakage-free
    construction — only uses information scouts could have known at
    draft time.

We then re-run the mover/stayer regression with HS-tier as a control,
under both methods, to see whether the mover coefficient survives once
HS quality is honestly accounted for.

Input:
  v3_analysis.csv

Outputs:
  v3_cohort_hs_tier_results.csv
"""

import pandas as pd
import numpy as np
import math
from scipy import stats
import statsmodels.api as sm
from config import START_YEAR, END_YEAR, COHORT_LABEL


df_all = pd.read_csv('v3_analysis.csv')
print(f"Loaded {len(df_all)} players")

hs_col = 'hs_name' if 'hs_name' in df_all.columns else 'school_name'
print(f"Using HS column: {hs_col}")

# Build a clean key: school name + state to disambiguate
df_all['hs_key'] = (df_all[hs_col].astype(str).str.strip()
                    + '|' + df_all['hs_state'].astype(str).str.strip())

# Restrict to cohort with all needed fields
df = df_all[(df_all['year'] >= START_YEAR) & (df_all['year'] <= END_YEAR)].copy()
df = df[df['mover'].notna() & df['distance_miles'].notna() & df['draftRound'].notna()].copy()
print(f"Cohort sample: {len(df)} players")

# ============================================================
# METHOD A: LEAVE-ONE-OUT HS MLB COUNT (full dataset)
# ============================================================
hs_total_mlb = df_all.groupby('hs_key')['reached_mlb'].sum().rename('hs_total_mlb')
hs_total_drafted = df_all.groupby('hs_key').size().rename('hs_total_drafted')
df = df.merge(hs_total_mlb, left_on='hs_key', right_index=True, how='left')
df = df.merge(hs_total_drafted, left_on='hs_key', right_index=True, how='left')

# Leave-one-out: subtract this player's own contribution
df['hs_loo_mlb']     = df['hs_total_mlb'] - df['reached_mlb']
df['hs_loo_drafted'] = df['hs_total_drafted'] - 1

# ============================================================
# METHOD B: PRIOR-HISTORY ONLY (drafts in years before this player's year)
# ============================================================
# For each (hs_key, year) compute cumulative MLB count from STRICTLY earlier years
df_all_sorted = df_all.sort_values(['hs_key', 'year']).copy()

# Build per-(hs_key, year) MLB and drafted counts
yearly = (df_all_sorted.groupby(['hs_key', 'year'])
          .agg(yr_mlb=('reached_mlb', 'sum'),
               yr_drafted=('reached_mlb', 'size'))
          .reset_index())
# Within each hs_key, compute cumulative sums up to but NOT including current year
yearly = yearly.sort_values(['hs_key', 'year']).reset_index(drop=True)
yearly['prior_mlb']     = yearly.groupby('hs_key')['yr_mlb'].cumsum().shift(1, fill_value=0)
yearly['prior_drafted'] = yearly.groupby('hs_key')['yr_drafted'].cumsum().shift(1, fill_value=0)
# Reset to 0 at start of each hs_key (the shift carries across groups incorrectly)
yearly.loc[yearly.groupby('hs_key').head(1).index, ['prior_mlb', 'prior_drafted']] = 0

# Merge back
df = df.merge(yearly[['hs_key', 'year', 'prior_mlb', 'prior_drafted']],
              on=['hs_key', 'year'], how='left')
df['prior_mlb']     = df['prior_mlb'].fillna(0).astype(int)
df['prior_drafted'] = df['prior_drafted'].fillna(0).astype(int)

# ============================================================
# Tier definitions (keep them coarse so we don't fragment the sample)
# ============================================================
def tier_loo(n):
    if pd.isna(n): return 'unknown'
    if n >= 5: return 't1_5plus'
    if n >= 2: return 't2_2_4'
    if n >= 1: return 't3_1'
    return 't4_zero'

df['hs_tier_loo']   = df['hs_loo_mlb'].apply(tier_loo)
df['hs_tier_prior'] = df['prior_mlb'].apply(tier_loo)

# Show MLB% by tier under both methods
print(f"\n{'='*75}")
print("MLB% BY HS TIER — LEAVE-ONE-OUT (excludes player himself)")
print(f"{'='*75}")
for t in ['t1_5plus', 't2_2_4', 't3_1', 't4_zero']:
    sub = df[df['hs_tier_loo'] == t]
    n = len(sub); m = int(sub['reached_mlb'].sum())
    pct = m/n*100 if n > 0 else 0
    print(f"  {t:<10}  N={n:>4}  MLB={m:>4}  MLB%={pct:>5.1f}%")

print(f"\n{'='*75}")
print("MLB% BY HS TIER — PRIOR HISTORY ONLY (years strictly before player's draft)")
print(f"{'='*75}")
for t in ['t1_5plus', 't2_2_4', 't3_1', 't4_zero']:
    sub = df[df['hs_tier_prior'] == t]
    n = len(sub); m = int(sub['reached_mlb'].sum())
    pct = m/n*100 if n > 0 else 0
    print(f"  {t:<10}  N={n:>4}  MLB={m:>4}  MLB%={pct:>5.1f}%")

# ============================================================
# Build regression features
# ============================================================
df['year_c'] = df['year'] - 2010
df['rounds_1_5']  = (df['draftRound'] <= 5).astype(int)
df['rounds_6_10'] = ((df['draftRound'] >= 6) & (df['draftRound'] <= 10)).astype(int)
df['rounds_11_20']= ((df['draftRound'] >= 11) & (df['draftRound'] <= 20)).astype(int)

# Tier dummies (LOO method)
df['loo_t1'] = (df['hs_tier_loo'] == 't1_5plus').astype(int)
df['loo_t2'] = (df['hs_tier_loo'] == 't2_2_4').astype(int)
df['loo_t3'] = (df['hs_tier_loo'] == 't3_1').astype(int)
# reference = t4_zero

# Tier dummies (PRIOR method)
df['prior_t1'] = (df['hs_tier_prior'] == 't1_5plus').astype(int)
df['prior_t2'] = (df['hs_tier_prior'] == 't2_2_4').astype(int)
df['prior_t3'] = (df['hs_tier_prior'] == 't3_1').astype(int)

# Also try HS MLB count as continuous (log-scaled)
df['log_hs_loo_mlb']   = np.log1p(df['hs_loo_mlb'])
df['log_hs_prior_mlb'] = np.log1p(df['prior_mlb'])

# ============================================================
# Run the models
# ============================================================
logit_rows = []

def run_model(label, feats):
    X = sm.add_constant(df[feats])
    y = df['reached_mlb']
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=400)
    except Exception as e:
        print(f"  {label} failed: {e}")
        return None
    print(f"\n--- {label} ---")
    print(f"  N={int(m.nobs)}  Pseudo R²={m.prsquared:.4f}  AIC={m.aic:.1f}")
    for var in X.columns:
        coef = m.params[var]; se = m.bse[var]; p = m.pvalues[var]; z = m.tvalues[var]
        odds = np.exp(coef)
        sig = ' ***' if p < 0.001 else (' **' if p < 0.01 else (' *' if p < 0.05 else ''))
        print(f"  {var:<22} coef={coef:+.4f}  OR={odds:8.3f}  SE={se:.4f}  z={z:+6.2f}  p={p:.4f}{sig}")
        logit_rows.append({
            'model': label, 'variable': var,
            'coef': round(coef, 6), 'odds_ratio': round(odds, 4),
            'std_err': round(se, 6), 'z': round(z, 3), 'p_value': round(p, 6),
            'n': int(m.nobs), 'pseudo_r2': round(m.prsquared, 5),
        })
    return m

base_feats = ['mover', 'year_c', 'rounds_1_5', 'rounds_6_10', 'rounds_11_20',
              'is_foreign_born', 'birth_warm_state', 'hs_warm_state']

# Baseline (no HS quality) for reference
print(f"\n{'='*80}")
print("REGRESSION RESULTS")
print(f"{'='*80}")
run_model('B1: BASELINE (no HS quality)', base_feats)

# LOO tier dummies
run_model('B2: + HS tier dummies (LEAVE-ONE-OUT)',
          base_feats + ['loo_t1', 'loo_t2', 'loo_t3'])

# LOO continuous
run_model('B3: + log(HS LOO MLB count)',
          base_feats + ['log_hs_loo_mlb'])

# PRIOR tier dummies
run_model('B4: + HS tier dummies (PRIOR HISTORY)',
          base_feats + ['prior_t1', 'prior_t2', 'prior_t3'])

# PRIOR continuous
run_model('B5: + log(HS PRIOR MLB count)',
          base_feats + ['log_hs_prior_mlb'])

# Both LOO + PRIOR continuous together (to see which dominates)
run_model('B6: + both LOO and PRIOR (continuous)',
          base_feats + ['log_hs_loo_mlb', 'log_hs_prior_mlb'])

pd.DataFrame(logit_rows).to_csv('v3_cohort_hs_tier_results.csv', index=False)
print(f"\nSaved: v3_cohort_hs_tier_results.csv")

# ============================================================
# Summary: how much does the mover coefficient shrink?
# ============================================================
print(f"\n{'='*80}")
print("MOVER COEFFICIENT — DOES HS QUALITY ABSORB IT?")
print(f"{'='*80}")
mover_rows = [r for r in logit_rows if r['variable'] == 'mover']
print(f"\n  {'Model':<45} {'Coef':>8} {'OR':>7} {'p':>10}")
for r in mover_rows:
    print(f"  {r['model']:<45} {r['coef']:>+8.4f} {r['odds_ratio']:>7.3f} {r['p_value']:>10.6f}")

print(f"\nInterpretation:")
print(f"  - If B2/B3/B4/B5 mover coef is close to B1, HS quality does NOT explain the mover effect.")
print(f"  - If it shrinks substantially, the mover effect was largely a proxy for HS quality.")
print(f"  - Compare LOO (B2/B3) vs PRIOR (B4/B5) -- PRIOR is the strict no-leakage test.")