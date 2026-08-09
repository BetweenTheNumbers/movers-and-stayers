"""
Draft-pedigree / "runway bias" probe.

Hypothesis: early-round picks get more minor-league runway (more chances,
slower release), so a mediocre early-round performer still gets pushed to MLB
while an equivalent late-round performer does not. Since movers are drafted
~7 rounds earlier, part of their higher MLB-attainment could be this promotion
advantage rather than talent.

We have NO minor-league performance data, so we cannot control for MiLB stats
directly. These two proxies triangulate instead:

TEST A -- Among MLB-reachers, does draft round predict career WAR?
  If early picks reach MLB but do NOT outperform late picks who also reached,
  that is consistent with promotion being pedigree-driven (runway) rather than
  merit-driven. Caveat: conditions on reaching MLB (survivorship), so read a
  flat/negative slope as suggestive, not proof.

TEST B -- Is the mover->MLB effect concentrated in EARLY rounds?
  Pure runway bias predicts the mover advantage is largest where runway is
  longest (early rounds). NOTE: the earlier late-round gradient found the
  OPPOSITE (mover OR biggest in deep rounds), so this test may COMPLICATE the
  runway story. Reported honestly either way.

Outputs:
  v3_runway_war_by_round.csv
  v3_runway_mover_by_roundband.csv

Run:  python scripts/v3_runway_bias.py
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
    import statsmodels.api as sm
    HAVE_SM = True
except ImportError:
    HAVE_SM = False


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
    print(f"Source: {src}   HS draftees {COHORT_LABEL}: {len(df):,}\n")
    return df


def band(r):
    if r <= 5:
        return '1-5'
    if r <= 10:
        return '6-10'
    if r <= 20:
        return '11-20'
    if r <= 40:
        return '21-40'
    return '41+'


def wilson_or(a, b, c, d):
    if min(a, b, c, d) == 0:
        a, b, c, d = a+0.5, b+0.5, c+0.5, d+0.5
    orr = (a*d)/(b*c)
    se = np.sqrt(1/a+1/b+1/c+1/d)
    return orr, np.exp(np.log(orr)-1.96*se), np.exp(np.log(orr)+1.96*se)


def main():
    df = load()

    # =================================================================
    # TEST A: among MLB-reachers, does draft round predict career WAR?
    # =================================================================
    print("=" * 82)
    print("TEST A -- Among MLB-reachers, does DRAFT ROUND predict career WAR?")
    print("=" * 82)
    print("If early picks reach MLB but don't outperform late picks who also")
    print("reached, promotion looks pedigree-driven (runway), not merit-driven.\n")

    reach = df[(df['reached_mlb'] == 1) & df['career_war'].notna()].copy()
    print(f"MLB-reachers with WAR: {len(reach):,}\n")

    rows = []
    print(f"{'Round band':<12} {'N':>5} {'mean WAR':>9} {'median':>8} "
          f"{'% >2 WAR':>9}")
    print("-" * 82)
    for b in ['1-5', '6-10', '11-20', '21-40', '41+']:
        sub = reach[reach['draftRound'].apply(band) == b]
        if len(sub) < 5:
            continue
        print(f"{b:<12} {len(sub):>5} {sub['career_war'].mean():>9.2f} "
              f"{sub['career_war'].median():>8.2f} "
              f"{(sub['career_war'] > 2).mean()*100:>8.1f}%")
        rows.append({'round_band': b, 'n': len(sub),
                     'mean_war': round(sub['career_war'].mean(), 3),
                     'median_war': round(sub['career_war'].median(), 3),
                     'pct_over_2war': round((sub['career_war'] > 2).mean(), 4)})
    pd.DataFrame(rows).to_csv('v3_runway_war_by_round.csv', index=False)

    # continuous test: WAR ~ round among reachers
    if HAVE_SM and len(reach) > 50:
        X = sm.add_constant(reach['draftRound'].astype(float))
        y = reach['career_war'].astype(float)
        res = sm.OLS(y, X).fit()
        slope = res.params['draftRound']
        pval = res.pvalues['draftRound']
        print(f"\nOLS: career WAR ~ draft round (among reachers)")
        print(f"  slope = {slope:+.4f} WAR per round, p = {pval:.4f}")
        if pval < 0.05 and slope < 0:
            print("  -> Later-round reachers DO perform worse: pedigree still")
            print("     tracks quality among survivors. Weaker runway signal.")
        elif pval >= 0.05:
            print("  -> Round does NOT predict WAR among reachers: once you make")
            print("     it, pedigree is uninformative. CONSISTENT with runway bias")
            print("     (early picks promoted despite equal eventual quality).")
    print()

    # =================================================================
    # TEST B: is the mover->MLB effect concentrated in early rounds?
    # =================================================================
    print("=" * 82)
    print("TEST B -- Is the MOVER->MLB effect concentrated in EARLY rounds?")
    print("=" * 82)
    print("Runway bias predicts the mover edge is LARGEST where runway is longest")
    print("(early rounds). Earlier work found the opposite (biggest in deep")
    print("rounds), so watch which way this points.\n")

    d = df[df['mover'].notna()].copy()
    d['mover'] = d['mover'].astype(int)
    d['band'] = d['draftRound'].apply(band)

    print(f"{'Round band':<12} {'Mv N':>6} {'Mv%MLB':>7} {'St N':>6} "
          f"{'St%MLB':>7} {'OR':>6} {'95% CI':>14}")
    print("-" * 82)
    brows = []
    for b in ['1-5', '6-10', '11-20', '21-40', '41+']:
        sub = d[d['band'] == b]
        mv = sub[sub['mover'] == 1]
        st = sub[sub['mover'] == 0]
        if len(mv) < 20 or len(st) < 20:
            continue
        a1, b1 = int(mv['reached_mlb'].sum()), int((1-mv['reached_mlb']).sum())
        c1, d1 = int(st['reached_mlb'].sum()), int((1-st['reached_mlb']).sum())
        orr, lo, hi = wilson_or(a1, b1, c1, d1)
        print(f"{b:<12} {len(mv):>6} {mv['reached_mlb'].mean()*100:>6.1f}% "
              f"{len(st):>6} {st['reached_mlb'].mean()*100:>6.1f}% "
              f"{orr:>6.2f} [{lo:>4.2f},{hi:>4.2f}]")
        brows.append({'round_band': b, 'mover_n': len(mv),
                      'mover_pct_mlb': round(mv['reached_mlb'].mean(), 4),
                      'stayer_n': len(st),
                      'stayer_pct_mlb': round(st['reached_mlb'].mean(), 4),
                      'odds_ratio': round(orr, 3), 'ci_lo': round(lo, 3),
                      'ci_hi': round(hi, 3)})
    pd.DataFrame(brows).to_csv('v3_runway_mover_by_roundband.csv', index=False)

    # formal interaction
    if HAVE_SM:
        d['early'] = (d['draftRound'] <= 10).astype(int)
        d['mover_x_early'] = d['mover'] * d['early']
        X = sm.add_constant(d[['mover', 'early', 'mover_x_early']])
        res = sm.Logit(d['reached_mlb'], X).fit(disp=False)
        ip = res.pvalues['mover_x_early']
        ic = res.params['mover_x_early']
        print(f"\nInteraction mover x (rounds 1-10):")
        print(f"  coef = {ic:+.4f}, OR = {np.exp(ic):.3f}, p = {ip:.4f}")
        if ip < 0.05 and ic > 0:
            print("  -> Mover effect STRONGER in early rounds: supports runway bias.")
        elif ip < 0.05 and ic < 0:
            print("  -> Mover effect WEAKER in early rounds (stronger late):")
            print("     AGAINST runway bias; matches the late-round gradient.")
        else:
            print("  -> No significant interaction: mover effect similar across")
            print("     early vs late. Runway bias not distinguished here.")

    print("\n" + "=" * 82)
    print("READING THE TWO TOGETHER")
    print("=" * 82)
    print("Runway bias is SUPPORTED if: (A) round doesn't predict WAR among")
    print("reachers, AND (B) the mover edge is concentrated early. If B instead")
    print("points late (as the earlier gradient suggested), the mechanism is more")
    print("likely thin LATE-round scouting coverage than early-round runway.")
    print("\nNOTE: no minor-league data here, so neither test controls MiLB")
    print("performance directly. Both are proxies; treat as suggestive.")
    print("\nSaved: v3_runway_war_by_round.csv, v3_runway_mover_by_roundband.csv")
    print("Done.")


if __name__ == '__main__':
    main()
