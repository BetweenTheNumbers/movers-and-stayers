"""
Step 19 — Is the mover effect changing over time?

The step-18 year-window test hinted the effect may be *weaker* in later
windows, which runs against the intuition that the travel-ball / showcase era
should amplify family-mobility advantages. But those windows are nested
subsets, so they cannot separate a real temporal trend from composition
changes. This tests it properly on one fixed sample.

Three complementary views:
  1. Pooled logistic with a mover x year interaction (the formal test).
  2. Era split: first half vs second half of the window.
  3. Per-year mover odds ratio with confidence intervals (the figure).

IMPORTANT CAVEATS printed with the output:
  - Draft length changed dramatically (100 rounds in 1996, ~50 through the
    2000s, 40 later). Later classes are a more selected pool, so round
    controls are included and the composition caveat still applies.
  - Later classes have had less time to reach MLB (right-censoring). This
    depresses the LEVEL of reach rates in recent years; it biases the mover
    GAP only if censoring differs by mover status, which is unlikely but
    cannot be ruled out.

Produces:
  v3_temporal_trend.csv          per-year odds ratios
  v3_temporal_trend_models.csv   interaction + era-split coefficients
  figures/fig21_mover_or_by_year.png

Input: v3_analysis.csv (or v3_analysis_with_war.csv)
Run:   python scripts/v3_temporal_trend.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Non-convergence is handled explicitly below; silence the duplicate warning.
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.simplefilter('ignore', ConvergenceWarning)
except ImportError:
    pass

try:
    import statsmodels.api as sm
except ImportError:
    print("ERROR: statsmodels required. pip install statsmodels")
    sys.exit(1)

from config import START_YEAR, END_YEAR, COHORT_LABEL

MIN_PER_YEAR = 60          # skip years with too few players for a stable OR
ROUND_FEATS = ['rounds_1_5', 'rounds_6_10', 'rounds_11_20']


def load():
    for f in ('v3_analysis_with_war.csv', 'v3_analysis.csv'):
        if os.path.exists(f):
            src = f
            break
    else:
        print("ERROR: no analysis CSV found. Run the pipeline first.")
        sys.exit(1)

    df = pd.read_csv(src, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[df['distance_miles'].notna() & df['mover'].notna()].copy()
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)]
    df['mover'] = df['mover'].astype(int)
    df['reached_mlb'] = df['reached_mlb'].astype(int)
    # centre year so the `mover` coefficient reads at mid-window, not year 0
    mid = (START_YEAR + END_YEAR) / 2.0
    df['year_c'] = df['year'] - mid
    df['rounds_1_5'] = (df['draftRound'] <= 5).astype(int)
    df['rounds_6_10'] = ((df['draftRound'] >= 6) & (df['draftRound'] <= 10)).astype(int)
    df['rounds_11_20'] = ((df['draftRound'] >= 11) & (df['draftRound'] <= 20)).astype(int)
    print(f"Source: {src}")
    print(f"Sample: {len(df):,} HS draftees with distance, {COHORT_LABEL}")
    print(f"Year centred at {mid:.1f}\n")
    return df, mid


def wilson_or_ci(a, b, c, d):
    """Odds ratio + 95% CI from a 2x2 (Woolf/log method). a,b = mover MLB/no."""
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    orr = (a * d) / (b * c)
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    return orr, np.exp(np.log(orr) - 1.96*se), np.exp(np.log(orr) + 1.96*se)


def main():
    df, mid = load()

    # ---- 1. Pooled interaction model ------------------------------------
    df['mover_x_year'] = df['mover'] * df['year_c']
    feats = ['mover', 'year_c', 'mover_x_year'] + ROUND_FEATS
    X = sm.add_constant(df[feats])
    y = df['reached_mlb']
    res = sm.Logit(y, X).fit(disp=False, maxiter=200)

    print("=" * 78)
    print("1. POOLED MODEL WITH mover x year INTERACTION")
    print("=" * 78)
    print(f"N={int(res.nobs):,}  Pseudo R2={res.prsquared:.4f}\n")
    model_rows = []
    for v in X.columns:
        if v == 'const':
            continue
        coef, p = res.params[v], res.pvalues[v]
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        tag = '  <<<' if v in ('mover', 'mover_x_year') else ''
        print(f"  {v:<16} coef={coef:+.4f}  OR={np.exp(coef):6.3f}  p={p:.4f} {sig}{tag}")
        model_rows.append({'model': 'interaction', 'variable': v,
                           'coef': round(coef, 5), 'odds_ratio': round(np.exp(coef), 4),
                           'p_value': p})

    ip = res.pvalues['mover_x_year']
    ic = res.params['mover_x_year']
    print(f"\n  Mover OR at mid-window ({mid:.0f}): {np.exp(res.params['mover']):.3f}")
    print(f"  Change per year: OR x {np.exp(ic):.4f}")
    if ip < 0.05:
        direction = "WEAKENING" if ic < 0 else "STRENGTHENING"
        print(f"  -> Interaction significant (p={ip:.4f}): the effect is {direction} "
              f"over time.")
    else:
        print(f"  -> Interaction NOT significant (p={ip:.4f}): no detectable trend; "
              f"the effect looks stable across the window.")

    # ---- 2. Era split ----------------------------------------------------
    split = int((START_YEAR + END_YEAR) // 2)
    print(f"\n{'=' * 78}")
    print(f"2. ERA SPLIT ({START_YEAR}-{split} vs {split+1}-{END_YEAR})")
    print("=" * 78)
    for lo, hi, label in [(START_YEAR, split, 'early'), (split + 1, END_YEAR, 'late')]:
        sub = df[(df['year'] >= lo) & (df['year'] <= hi)]
        if len(sub) < 200:
            continue
        Xe = sm.add_constant(sub[['mover', 'year_c'] + ROUND_FEATS])
        re_ = sm.Logit(sub['reached_mlb'], Xe).fit(disp=False, maxiter=200)
        coef, se = re_.params['mover'], re_.bse['mover']
        orr = np.exp(coef)
        lo_ci, hi_ci = np.exp(coef - 1.96*se), np.exp(coef + 1.96*se)
        mv = sub[sub['mover'] == 1]['reached_mlb'].mean() * 100
        st = sub[sub['mover'] == 0]['reached_mlb'].mean() * 100
        print(f"\n  {lo}-{hi} (n={len(sub):,})")
        print(f"    Movers {mv:.1f}% vs stayers {st:.1f}%  (gap {mv-st:+.1f} pp)")
        print(f"    Mover OR {orr:.3f} [{lo_ci:.2f}, {hi_ci:.2f}]  p={re_.pvalues['mover']:.2e}")
        model_rows.append({'model': f'era_{label}', 'variable': 'mover',
                           'coef': round(coef, 5), 'odds_ratio': round(orr, 4),
                           'p_value': re_.pvalues['mover']})

    pd.DataFrame(model_rows).to_csv('v3_temporal_trend_models.csv', index=False)

    # ---- 2b. Round-restricted interaction --------------------------------
    # The draft shrank from 100 rounds (1996) to 50 (1997-2011) to 40 (2012+).
    # Deep rounds are full of marginal players, so cutting them mechanically
    # raises average reach rates. If stayers were overrepresented in those
    # rounds, the apparent decline in the mover effect is composition, not a
    # real trend. Restricting every year to a common round window removes it.
    print(f"\n{'=' * 78}")
    print("2b. ROUND-RESTRICTED INTERACTION (comparable pools across years)")
    print("=" * 78)
    print("If the interaction survives here, the trend is not a draft-length artifact.\n")
    for max_round in (40, 20, 10):
        sub = df[df['draftRound'] <= max_round].copy()
        if len(sub) < 500 or sub['reached_mlb'].nunique() < 2:
            continue
        # keep only predictors that actually vary in this subsample
        feats_r = [f for f in (['mover', 'year_c', 'mover_x_year'] + ROUND_FEATS)
                   if sub[f].nunique() > 1]
        Xr = sm.add_constant(sub[feats_r])
        try:
            rr = sm.Logit(sub['reached_mlb'], Xr).fit(disp=False, maxiter=300)
        except Exception as e:
            print(f"  Rounds 1-{max_round:<3} model failed to fit ({e.__class__.__name__})")
            continue
        if not getattr(rr, 'mle_retvals', {}).get('converged', True):
            print(f"  Rounds 1-{max_round:<3} n={len(sub):>6,}  "
                  f"(did not converge — too few events, skipping)")
            continue
        ic_r, ip_r = rr.params['mover_x_year'], rr.pvalues['mover_x_year']
        verdict = 'still declining' if (ip_r < 0.05 and ic_r < 0) else \
                  ('no trend' if ip_r >= 0.05 else 'increasing')
        print(f"  Rounds 1-{max_round:<3} n={len(sub):>6,}  "
              f"mover OR(mid)={np.exp(rr.params['mover']):.3f}  "
              f"interaction={np.exp(ic_r):.4f}/yr  p={ip_r:.4f}  -> {verdict}")
        model_rows.append({'model': f'rounds_1_{max_round}',
                           'variable': 'mover_x_year',
                           'coef': round(ic_r, 5),
                           'odds_ratio': round(np.exp(ic_r), 4),
                           'p_value': ip_r})

    # ---- 2c. Where did the stayer gain come from? ------------------------
    # The gap closed because stayers improved, not because movers declined.
    # Show reach rate by era WITHIN a fixed round band to see if that holds.
    print(f"\n{'=' * 78}")
    print("2c. IS THE STAYER GAIN CONCENTRATED IN THE DEEP ROUNDS?")
    print("=" * 78)
    print(f"{'Round band':<14} {'Era':<12} {'Movers':>8} {'Stayers':>9} {'Gap':>7} {'N':>7}")
    print("-" * 78)
    bands = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 200)]
    for lo_r, hi_r in bands:
        for lo_y, hi_y, lab in [(START_YEAR, split, f'{START_YEAR}-{split}'),
                                (split + 1, END_YEAR, f'{split+1}-{END_YEAR}')]:
            sub = df[(df['draftRound'] >= lo_r) & (df['draftRound'] <= hi_r) &
                     (df['year'] >= lo_y) & (df['year'] <= hi_y)]
            if len(sub) < 50:
                continue
            mv = sub[sub['mover'] == 1]['reached_mlb']
            st = sub[sub['mover'] == 0]['reached_mlb']
            if len(mv) == 0 or len(st) == 0:
                continue
            band = f"{lo_r}-{hi_r if hi_r < 200 else '+'}"
            print(f"{band:<14} {lab:<12} {mv.mean()*100:>7.1f}% {st.mean()*100:>8.1f}% "
                  f"{(mv.mean()-st.mean())*100:>+6.1f} {len(sub):>7,}")

    pd.DataFrame(model_rows).to_csv('v3_temporal_trend_models.csv', index=False)

    # ---- 3. Per-year odds ratios ----------------------------------------
    print(f"\n{'=' * 78}")
    print("3. MOVER ODDS RATIO BY DRAFT YEAR")
    print("=" * 78)
    print(f"{'Year':>6} {'N':>6} {'Mover%':>8} {'Stay%':>8} {'OR':>7} {'95% CI':>16}")
    print("-" * 78)
    rows = []
    for yr in sorted(df['year'].unique()):
        sub = df[df['year'] == yr]
        if len(sub) < MIN_PER_YEAR:
            continue
        a = int(((sub['mover'] == 1) & (sub['reached_mlb'] == 1)).sum())
        b = int(((sub['mover'] == 1) & (sub['reached_mlb'] == 0)).sum())
        c = int(((sub['mover'] == 0) & (sub['reached_mlb'] == 1)).sum())
        d = int(((sub['mover'] == 0) & (sub['reached_mlb'] == 0)).sum())
        if min(a + b, c + d) == 0:
            continue
        orr, lo_ci, hi_ci = wilson_or_ci(a, b, c, d)
        mv = a / (a + b) * 100
        st = c / (c + d) * 100
        print(f"{yr:>6} {len(sub):>6,} {mv:>7.1f}% {st:>7.1f}% {orr:>7.2f} "
              f"[{lo_ci:>5.2f}, {hi_ci:>5.2f}]")
        rows.append({'year': yr, 'n': len(sub), 'mover_pct': round(mv, 2),
                     'stayer_pct': round(st, 2), 'odds_ratio': round(orr, 4),
                     'ci_low': round(lo_ci, 4), 'ci_high': round(hi_ci, 4)})

    per_year = pd.DataFrame(rows)
    per_year.to_csv('v3_temporal_trend.csv', index=False)
    print(f"\nSaved: v3_temporal_trend.csv, v3_temporal_trend_models.csv")

    print(f"\n{'=' * 78}")
    print("CAVEATS")
    print("=" * 78)
    print("  * Draft length changed over the window (100 rounds in 1996, ~50 in the")
    print("    2000s, 40 later). Later classes are a more selected pool. Round")
    print("    controls are included, but composition effects may remain.")
    print("  * Later classes have had less time to reach MLB. This lowers the LEVEL")
    print("    of reach rates in recent years. It biases the mover GAP only if")
    print("    censoring differs by mover status.")

    # ---- figure ----------------------------------------------------------
    if len(per_year) >= 3:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams.update({'figure.dpi': 150, 'savefig.dpi': 150,
                                 'font.size': 12, 'axes.titlesize': 15,
                                 'axes.titleweight': 'bold',
                                 'axes.spines.top': False, 'axes.spines.right': False,
                                 'axes.grid': True, 'grid.alpha': 0.25})
            os.makedirs('figures', exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 6.5))
            yrs = per_year['year'].values
            ors = per_year['odds_ratio'].values
            lo_e = ors - per_year['ci_low'].values
            hi_e = per_year['ci_high'].values - ors
            ax.errorbar(yrs, ors, yerr=[lo_e, hi_e], fmt='o', color='#2C6FBB',
                        markersize=6, capsize=3, linewidth=1, alpha=0.85,
                        label='Mover OR by year (95% CI)')
            # fitted trend from the interaction model
            grid = np.linspace(yrs.min(), yrs.max(), 100)
            fitted = np.exp(res.params['mover'] + res.params['mover_x_year'] * (grid - mid))
            ax.plot(grid, fitted, color='#C44E52', lw=2.5,
                    label=f'Fitted trend (interaction p={ip:.3f})')
            ax.axhline(1.0, ls='--', color='#555', lw=1)
            ax.text(yrs.max(), 1.02, 'OR = 1 (no effect)', ha='right', va='bottom',
                    fontsize=9, color='#555', style='italic')
            ax.set_xlabel('Draft year')
            ax.set_ylabel('Mover odds ratio (reaching MLB)')
            ax.set_title('Is the mover effect changing over time?')
            ax.legend(frameon=False, loc='upper right')
            fig.text(0.99, 0.01,
                     f'HS draftees {COHORT_LABEL} (n={len(df):,}). Years with fewer than '
                     f'{MIN_PER_YEAR} players omitted. Wide CIs reflect small yearly samples.',
                     ha='right', fontsize=8, color='#555')
            fig.tight_layout()
            p = 'figures/fig21_mover_or_by_year.png'
            fig.savefig(p, bbox_inches='tight'); plt.close(fig)
            print(f"  saved {p}")
        except ImportError:
            print("  (matplotlib unavailable; figure skipped)")
    print("\nDone.")


if __name__ == '__main__':
    main()
