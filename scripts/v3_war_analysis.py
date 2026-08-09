"""
Mover/Stayer analysis on FanGraphs career outcomes.

For each mover/stayer subgroup, we now compare:
  - Reached MLB at all (the binary we had)
  - Reached 1+ WAR career   (genuine MLB player)
  - Reached 3+ WAR career   (multi-year contributor)
  - Reached 10+ WAR career  (quality long career)
  - Reached 25+ WAR career  (All-Star / borderline HOF)
  - Career WAR (continuous, conditional on reaching MLB)
  - WAR per game (efficiency, conditional on reaching MLB)
  - Career games (longevity, conditional on reaching MLB)

The key question: is the mover advantage purely a "got to MLB" effect, or
does it persist into "had a good MLB career"?

Two interpretations possible:
  A. Mover effect is ONLY at the binary-MLB threshold -> the mover advantage
     is about being scouted/exposed, not about underlying talent. Once a
     player reaches MLB, his career quality is independent of mover status.
  B. Mover effect SCALES with career quality -> movers are not just more
     likely to be scouted, they're also genuinely better players. This is
     consistent with a family-resources / parental-investment confounder
     that affects development quality, not just exposure.

Input:
  v3_analysis_with_war.csv

Outputs:
  v3_war_outcomes_by_mover.csv
  v3_war_by_distance_bin.csv
  v3_war_logit.csv
"""

import pandas as pd
import numpy as np
import math
import os
import time
from scipy import stats
import statsmodels.api as sm
from config import START_YEAR, END_YEAR, COHORT_LABEL


def safe_to_csv(df_out, path):
    """Write a CSV, but if the target is locked (open in Excel) fall back to a
    timestamped filename instead of failing and leaving a 0-KB stub."""
    try:
        df_out.to_csv(path, index=False)
        print(f"  saved {path} ({len(df_out)} rows)")
        return
    except (PermissionError, OSError) as e:
        base, ext = os.path.splitext(path)
        alt = f"{base}_{time.strftime('%H%M%S')}{ext}"
        df_out.to_csv(alt, index=False)
        print(f"  WARNING: {path} was locked ({e.__class__.__name__}); "
              f"wrote {alt} instead. Close it in Excel and re-run to refresh the main file.")


df_all = pd.read_csv('v3_analysis_with_war.csv')
print(f"Loaded {len(df_all)} players")

# Focus on the cohort where everyone has had time to develop
df = df_all[(df_all['year'] >= START_YEAR) & (df_all['year'] <= END_YEAR)].copy()
print(f"{START_YEAR}-{END_YEAR} cohort: {len(df)}")

# Players who reached MLB AND have FG match (for conditional analyses)
mlb = df[(df['reached_mlb'] == 1) & (df['fg_any_match'] == 1)].copy()
print(f"  Reached MLB with FG match: {len(mlb)}")
print(f"  Movers among these:        {(mlb['mover'] == 1).sum()}")
print(f"  Stayers among these:       {(mlb['mover'] == 0).sum()}")


# ============================================================
# 1. BINARY-OUTCOME COMPARISON: mover vs stayer at each threshold
# ============================================================
print(f"\n{'='*85}")
print(f"MOVER vs STAYER at each career-quality threshold ({COHORT_LABEL} cohort)")
print(f"{'='*85}")

# For each threshold, compute (1) the unconditional rate (% of ALL HS draftees
# who hit the threshold) and (2) the chi-squared p-value.

outcome_cols = [
    ('reached_mlb',       'Reached MLB',         None),
    ('had_1war_career',   '1+ WAR career',       1.0),
    ('had_3war_career',   '3+ WAR career',       3.0),
    ('had_10war_career',  '10+ WAR career',      10.0),
    ('had_25war_career',  '25+ WAR career',      25.0),
]

# For WAR-tier outcomes, we need to make sure NaN (no FG match) is treated as 0
# for "did NOT hit the threshold" — but only among reached_mlb=0 players.
# A reached_mlb=1 with no FG match (29 players) is ambiguous; we treat them
# as "reached MLB but career too small to measure" -> below threshold.
for col, _, _ in outcome_cols:
    if col == 'reached_mlb':
        continue
    df[col] = df[col].fillna(0)

print(f"{'Outcome':<18} {'N Stay':>6} {'Stay%':>7} {'N Move':>7} {'Move%':>7} "
      f"{'Diff':>7} {'Ratio':>6} {'p':>10}")
print('-' * 85)

outcome_rows = []
base = df[df['mover'].notna()]

for col, label, _ in outcome_cols:
    s = base[base['mover'] == 0]
    m = base[base['mover'] == 1]
    s_rate = s[col].mean() * 100
    m_rate = m[col].mean() * 100
    diff = m_rate - s_rate
    ratio = m_rate / s_rate if s_rate > 0 else float('nan')

    ct = pd.crosstab(base['mover'], base[col])
    p_val = float('nan')
    if ct.shape == (2, 2):
        _, p_val, _, _ = stats.chi2_contingency(ct)

    print(f"{label:<18} {len(s):>6} {s_rate:>6.2f}% {len(m):>7} {m_rate:>6.2f}% "
          f"{diff:>+6.2f} {ratio:>6.2f} {p_val:>10.6f}")
    outcome_rows.append({
        'outcome': col, 'label': label,
        'n_stayers': len(s), 'n_movers': len(m),
        'stayer_rate_pct': round(s_rate, 3),
        'mover_rate_pct': round(m_rate, 3),
        'diff_pp': round(diff, 3),
        'mover_to_stayer_ratio': round(ratio, 3),
        'p_value': round(p_val, 8) if not math.isnan(p_val) else None,
    })

safe_to_csv(pd.DataFrame(outcome_rows), 'v3_war_outcomes_by_mover.csv')


# ============================================================
# 2. CONDITIONAL ANALYSIS — among MLB-reachers, do movers have better careers?
# ============================================================
print(f"\n{'='*85}")
print("AMONG MLB-REACHERS: does career quality differ by mover status?")
print(f"{'='*85}")

mlb_base = mlb[mlb['mover'].notna()].copy()
s = mlb_base[mlb_base['mover'] == 0]
m = mlb_base[mlb_base['mover'] == 1]
print(f"\nSample sizes: stayers={len(s)}, movers={len(m)}")

continuous_outcomes = [
    ('career_war',    'Career WAR'),
    ('war_per_game',  'WAR / game'),
    ('career_games',  'Career games'),
    ('hit_pa',        'Hitting PA (NaN excluded)'),
    ('pit_ip',        'Pitching IP (NaN excluded)'),
]

print(f"\n{'Outcome':<28} {'Stayer mean':>12} {'Mover mean':>12} {'Diff':>9} {'t-test p':>10}")
print('-' * 80)

for col, label in continuous_outcomes:
    s_vals = s[col].dropna()
    m_vals = m[col].dropna()
    if len(s_vals) < 5 or len(m_vals) < 5:
        continue
    s_mean = s_vals.mean()
    m_mean = m_vals.mean()
    t_stat, p_val = stats.ttest_ind(s_vals, m_vals, equal_var=False)
    print(f"{label:<28} {s_mean:>12.3f} {m_mean:>12.3f} "
          f"{m_mean - s_mean:>+8.3f} {p_val:>10.6f}")

# Also percentile breakdown
print(f"\n{'Percentile':<6} {'Career WAR — Stayers':>22} {'Career WAR — Movers':>22}")
print('-' * 55)
for q in [25, 50, 75, 90, 95, 99]:
    s_q = s['career_war'].dropna().quantile(q/100)
    m_q = m['career_war'].dropna().quantile(q/100)
    print(f"P{q:<5d} {s_q:>22.2f} {m_q:>22.2f}")


# ============================================================
# 3. WAR OUTCOMES BY DISTANCE BIN — does career quality scale with distance?
# ============================================================
print(f"\n{'='*100}")
print(f"CAREER OUTCOMES BY DISTANCE BIN ({COHORT_LABEL} cohort)")
print(f"{'='*100}")
print(f"{'Bin':<14} {'N':>5} "
      f"{'%MLB':>6} {'%1WAR':>7} {'%3WAR':>7} {'%10WAR':>8} {'%25WAR':>8} "
      f"{'mean WAR (MLB)':>16}")
print('-' * 95)

bin_order = ['00_exact', '01_under5', '02_5_10', '03_10_15', '04_15_20', '05_20_25',
             '06_25_50', '07_50_100', '08_100_250', '09_250_500', '10_500_1000',
             '11_1000_plus']

dist_rows = []
for b in bin_order:
    sub = df[df['distance_bin'] == b]
    n = len(sub)
    if n == 0: continue
    pct_mlb     = sub['reached_mlb'].mean() * 100
    pct_1war    = sub['had_1war_career'].mean() * 100
    pct_3war    = sub['had_3war_career'].mean() * 100
    pct_10war   = sub['had_10war_career'].mean() * 100
    pct_25war   = sub['had_25war_career'].mean() * 100
    mlb_subset  = sub[sub['fg_any_match'] == 1]
    mean_war    = mlb_subset['career_war'].mean() if len(mlb_subset) > 0 else float('nan')

    print(f"{b:<14} {n:>5} "
          f"{pct_mlb:>5.1f}% {pct_1war:>6.1f}% {pct_3war:>6.1f}% "
          f"{pct_10war:>7.1f}% {pct_25war:>7.1f}% {mean_war:>15.2f}")
    dist_rows.append({
        'distance_bin': b, 'n': n,
        'pct_mlb': round(pct_mlb, 2),
        'pct_1war': round(pct_1war, 2),
        'pct_3war': round(pct_3war, 2),
        'pct_10war': round(pct_10war, 2),
        'pct_25war': round(pct_25war, 2),
        'mean_career_war_among_mlb': round(mean_war, 3) if not math.isnan(mean_war) else None,
    })

safe_to_csv(pd.DataFrame(dist_rows), 'v3_war_by_distance_bin.csv')


# ============================================================
# 4. LOGISTIC REGRESSION FOR EACH OUTCOME
# ============================================================
print(f"\n{'='*90}")
print("LOGISTIC REGRESSION — mover effect on each outcome (with full controls)")
print(f"{'='*90}")

model_df = df[df['mover'].notna() & df['distance_miles'].notna() & df['draftRound'].notna()].copy()
print(f"Sample: {len(model_df)} players")

model_df['year_c'] = model_df['year'] - 2010
model_df['rounds_1_5']  = (model_df['draftRound'] <= 5).astype(int)
model_df['rounds_6_10'] = ((model_df['draftRound'] >= 6) & (model_df['draftRound'] <= 10)).astype(int)
model_df['rounds_11_20']= ((model_df['draftRound'] >= 11) & (model_df['draftRound'] <= 20)).astype(int)

base_feats = ['mover', 'year_c', 'rounds_1_5', 'rounds_6_10', 'rounds_11_20',
              'is_foreign_born', 'birth_warm_state', 'hs_warm_state']

logit_rows = []

def run_logit(label, outcome):
    sub = model_df[model_df[outcome].notna()].copy()
    if sub[outcome].nunique() < 2:
        print(f"\n--- {label}: insufficient variation, skipped")
        return
    X = sm.add_constant(sub[base_feats])
    y = sub[outcome].astype(int)
    try:
        m = sm.Logit(y, X).fit(disp=False, maxiter=200)
    except Exception as e:
        print(f"\n--- {label} failed: {e}")
        return
    print(f"\n--- {label}  (outcome: {outcome}) ---")
    print(f"  N={int(m.nobs)}  PosRate={y.mean()*100:.1f}%  Pseudo R²={m.prsquared:.4f}")
    for var in X.columns:
        if var == 'const': continue
        coef = m.params[var]; se = m.bse[var]; p = m.pvalues[var]; z = m.tvalues[var]
        odds = np.exp(coef)
        sig = ' ***' if p < 0.001 else (' **' if p < 0.01 else (' *' if p < 0.05 else ''))
        if var == 'mover':
            print(f"  {var:<22} coef={coef:+.4f}  OR={odds:6.3f}  p={p:.4f}{sig}  <<<")
        else:
            print(f"  {var:<22} coef={coef:+.4f}  OR={odds:6.3f}  p={p:.4f}{sig}")
        logit_rows.append({
            'outcome': outcome, 'model': label,
            'variable': var, 'coef': round(coef, 6),
            'odds_ratio': round(odds, 4), 'std_err': round(se, 6),
            'z': round(z, 3), 'p_value': round(p, 6),
            'n': int(m.nobs), 'pseudo_r2': round(m.prsquared, 5),
            'positive_rate': round(y.mean(), 4),
        })

run_logit('Outcome: Reached MLB',  'reached_mlb')
run_logit('Outcome: 1+ WAR career', 'had_1war_career')
run_logit('Outcome: 3+ WAR career', 'had_3war_career')
run_logit('Outcome: 10+ WAR career','had_10war_career')
run_logit('Outcome: 25+ WAR career','had_25war_career')

safe_to_csv(pd.DataFrame(logit_rows), 'v3_war_logit.csv')


# ============================================================
# 5. SUMMARY: mover odds ratio across outcomes
# ============================================================
print(f"\n{'='*85}")
print("MOVER ODDS RATIO ACROSS OUTCOMES (controlled model)")
print(f"{'='*85}")
print(f"{'Outcome':<22} {'PosRate':>8} {'Mover OR':>10} {'p':>10}")
print('-' * 60)

mover_rows = [r for r in logit_rows if r['variable'] == 'mover']
# Order by stringency
order_keys = ['reached_mlb', 'had_1war_career', 'had_3war_career',
              'had_10war_career', 'had_25war_career']
for k in order_keys:
    rec = next((r for r in mover_rows if r['outcome'] == k), None)
    if rec is None: continue
    print(f"{k:<22} {rec['positive_rate']*100:>6.2f}% {rec['odds_ratio']:>10.3f} "
          f"{rec['p_value']:>10.6f}")

print(f"\nInterpretation:")
print(f"  - If mover OR is roughly CONSTANT across outcomes, the effect is uniform")
print(f"    (movers are equally advantaged at reaching MLB and at career quality).")
print(f"  - If mover OR INCREASES with stringency, the effect strengthens at the elite")
print(f"    end (movers are disproportionately the elite players).")
print(f"  - If mover OR DECREASES with stringency, the effect is concentrated at the")
print(f"    binary-MLB threshold (movers get to MLB more, but aren't better players).")