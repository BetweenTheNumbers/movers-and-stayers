"""
Mover-tier discovery and testing.

STAGE 1 -- find where the mover-distance distribution naturally breaks, using
three independent methods. If they agree, the tiers are real; if not, distance
is a smooth continuum and any cut is a presentation choice (also a finding).
  (a) density histogram / log-distance modes
  (b) Jenks natural breaks (1-D variance minimization, the standard)
  (c) k-means sweep k=2..6 with silhouette, to see if any k is preferred

STAGE 2 -- with the chosen tiers, test each against THREE outcomes:
  1. MLB reach rate (arrival)          -- all draftees
  2. value RATE among reachers         -- WAR/600 PA (hitters), WAR/150 IP (pitchers)  [FanGraphs house rate]
  3. playing time among reachers       -- games, PA, IP
Round-normalized two ways: descriptive bands (1-10, 11-20, 21+) AND a logit/OLS
with round as a covariate plus tier dummies.

Stayers (distance 0 / same city) are always their own baseline tier.

Outputs:
  v3_tier_breaks.csv          the discovered breakpoints from each method
  v3_tier_arrival.csv         reach-rate by tier, overall + by round band
  v3_tier_value.csv           value-rate + playing-time by tier (reachers)
  v3_tier_model.csv           logit (arrival) + OLS (value) coefficients

Run:  python scripts/v3_mover_tiers.py
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
    import statsmodels.formula.api as smf
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

PA_QUAL = 145   # ~130 AB
IP_QUAL = 50
SAME_CITY_MAX = 1.0   # miles; at/under this = stayer (geocode jitter tolerance)


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
    if dcol is None:
        print("ERROR: no distance column found.")
        sys.exit(1)
    df = df[df[dcol].notna()].copy()
    df = df.rename(columns={dcol: 'dist'})
    print(f"Source: {src}   HS draftees w/ distance {COHORT_LABEL}: {len(df):,}")
    return df


def discover_breaks(dist_movers):
    """Three methods to find natural breakpoints in mover distances (miles>SAME_CITY_MAX)."""
    rows = []
    x = dist_movers.values
    logx = np.log10(x)

    # (a) log-distance histogram modes (report decile edges as a readable summary)
    qs = [10, 25, 50, 75, 90]
    perc = np.percentile(x, qs)
    for q, p in zip(qs, perc):
        rows.append({'method': 'percentile', 'k_or_q': q, 'break_mi': round(p, 1)})

    # (b) Jenks natural breaks (pure-numpy Fisher-Jenks for 1-D)
    def jenks(data, n_classes):
        data = np.sort(data)
        # dynamic programming Fisher-Jenks
        mat1 = np.zeros((len(data)+1, n_classes+1))
        mat2 = np.zeros((len(data)+1, n_classes+1))
        mat1[1, 1:] = 1
        mat2[2:, 1:] = np.inf
        v = 0.0
        for l in range(2, len(data)+1):
            s1 = s2 = w = 0.0
            for m in range(1, l+1):
                i3 = l - m + 1
                val = data[i3-1]
                s2 += val*val; s1 += val; w += 1
                v = s2 - (s1*s1)/w
                i4 = i3 - 1
                if i4 != 0:
                    for j in range(2, n_classes+1):
                        if mat2[l, j] >= (v + mat2[i4, j-1]):
                            mat1[l, j] = i3
                            mat2[l, j] = v + mat2[i4, j-1]
            mat1[l, 1] = 1
            mat2[l, 1] = v
        k = len(data)
        kclass = [0]*(n_classes+1)
        kclass[n_classes] = data[-1]
        kclass[0] = data[0]
        cnt = n_classes
        while cnt >= 2:
            idx = int(mat1[k, cnt]) - 2
            kclass[cnt-1] = data[idx]
            k = int(mat1[k, cnt]) - 1
            cnt -= 1
        return kclass
    try:
        # sample for speed if large
        samp = x if len(x) <= 4000 else np.random.RandomState(0).choice(x, 4000, replace=False)
        for nc in (3, 4):
            br = jenks(samp, nc)
            for b in br[1:-1]:
                rows.append({'method': f'jenks_{nc}', 'k_or_q': nc, 'break_mi': round(b, 1)})
    except Exception as e:
        rows.append({'method': 'jenks', 'k_or_q': 0, 'break_mi': f'failed:{e}'})

    # (c) k-means on log-distance, silhouette to pick k
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best = None
        for k in range(2, 7):
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(logx.reshape(-1, 1))
            sil = silhouette_score(logx.reshape(-1, 1), km.labels_)
            rows.append({'method': 'kmeans_silhouette', 'k_or_q': k,
                         'break_mi': round(sil, 3)})
            if best is None or sil > best[1]:
                best = (k, sil, km)
        # report the boundaries of the best-k solution
        k, sil, km = best
        centers = np.sort(km.cluster_centers_.ravel())
        bounds = 10**((centers[:-1] + centers[1:]) / 2)
        for b in bounds:
            rows.append({'method': f'kmeans_best_k{k}', 'k_or_q': k,
                         'break_mi': round(b, 1)})
        print(f"  k-means best silhouette at k={k} ({sil:.3f})")
    except ImportError:
        rows.append({'method': 'kmeans', 'k_or_q': 0, 'break_mi': 'sklearn missing'})

    return pd.DataFrame(rows)


def assign_tier(d, edges):
    """edges e.g. [50, 500] -> Stayer / Short / Medium / Long."""
    if d <= SAME_CITY_MAX:
        return '0 Stayer'
    names = ['1 Short', '2 Medium', '3 Long', '4 XLong']
    for i, e in enumerate(edges):
        if d <= e:
            return names[i]
    return names[len(edges)]


def band(r):
    if r <= 10:
        return '1-10'
    if r <= 20:
        return '11-20'
    return '21+'


def main():
    df = load()
    movers = df[df['dist'] > SAME_CITY_MAX]
    print(f"Movers (> {SAME_CITY_MAX}mi): {len(movers):,}   "
          f"Stayers: {(df['dist'] <= SAME_CITY_MAX).sum():,}\n")

    # ---- STAGE 1: discover breaks ----
    print("="*78)
    print("STAGE 1  discovering natural distance breaks among movers")
    print("="*78)
    breaks = discover_breaks(movers['dist'])
    breaks.to_csv('v3_tier_breaks.csv', index=False)
    print(breaks.to_string(index=False))
    print("\nInterpretation: if jenks/kmeans/percentile roughly agree on cut points,")
    print("tiers are real. If not, distance is smooth (report as continuum).\n")

    # choose edges: use rounded, defensible values informed by the breaks above.
    # (kept fixed & interpretable so the talk survives a methods question)
    # NOTE: on real data k-means preferred k=2 (no clean natural break), so these
    # tiers are a *presentation* choice over a smooth continuum, not discovered
    # clusters. The binary mover/stayer split is always the safe fallback.
    edges = [50, 500]   # Short <=50, Medium 50-500, Long >500
    print(f"Using interpretable tier edges (mi): {edges}  "
          f"-> Stayer / Short / Medium / Long")
    print("(distance is ~continuous; tiers are a readable cut, not natural clusters)\n")
    df['tier'] = df['dist'].apply(lambda d: assign_tier(d, edges))
    df['band'] = df['draftRound'].apply(band)

    # ---- STAGE 2a: arrival by tier ----
    print("="*78)
    print("STAGE 2a  MLB reach rate by tier (overall + round bands)")
    print("="*78)
    arr_rows = []
    order = ['0 Stayer', '1 Short', '2 Medium', '3 Long']
    print(f"{'tier':<10} {'N':>6} {'MLB%':>7}   by round band ->")
    for t in order:
        sub = df[df['tier'] == t]
        if len(sub) == 0:
            continue
        line = f"{t:<10} {len(sub):>6} {sub['reached_mlb'].mean()*100:>6.1f}%   "
        rec = {'tier': t, 'n': len(sub),
               'mlb_rate': round(sub['reached_mlb'].mean(), 4)}
        for b in ['1-10', '11-20', '21+']:
            bs = sub[sub['band'] == b]
            if len(bs) >= 20:
                line += f"{b}:{bs['reached_mlb'].mean()*100:4.1f}% "
                rec[f'rate_{b}'] = round(bs['reached_mlb'].mean(), 4)
        print(line)
        arr_rows.append(rec)
    pd.DataFrame(arr_rows).to_csv('v3_tier_arrival.csv', index=False)

    # ---- STAGE 2b: value rate + playing time among reachers ----
    print("\n" + "="*78)
    print("STAGE 2b  value rate + playing time by tier (MLB reachers only)")
    print("="*78)
    r = df[(df['reached_mlb'] == 1)].copy()
    # map the FanGraphs-join column names (hit_pa/pit_ip/*_games) to what we need
    rename = {}
    if 'hit_pa' in r.columns:
        rename['hit_pa'] = 'career_pa'
    if 'pit_ip' in r.columns:
        rename['pit_ip'] = 'career_ip'
    r = r.rename(columns=rename)
    if 'career_games' in r.columns:
        r['career_g'] = r['career_games']
    elif {'hit_games', 'pit_games'} & set(r.columns):
        r['career_g'] = r[['hit_games', 'pit_games']].max(axis=1)
    for c in ['career_pa', 'career_ip', 'career_g', 'career_war']:
        if c not in r.columns:
            r[c] = np.nan
    r['is_pitcher'] = (r['career_ip'].fillna(0) * 4.3 >= r['career_pa'].fillna(0)).astype(int)
    # per-unit rates must use the COMPONENT war, not combined career_war:
    # hitting WAR over PA, pitching WAR over IP. Falls back to career_war only
    # if the component column is absent.
    hw = r['hit_war'] if 'hit_war' in r.columns else r['career_war']
    pw = r['pit_war'] if 'pit_war' in r.columns else r['career_war']
    r['war_per_600pa'] = np.where(r['career_pa'] >= PA_QUAL,
                                  hw / r['career_pa'] * 600, np.nan)
    r['war_per_150ip'] = np.where(r['career_ip'] >= IP_QUAL,
                                  pw / r['career_ip'] * 150, np.nan)

    val_rows = []
    print(f"{'tier':<10} {'N':>5} {'WAR':>7} {'G':>6} {'PA':>7} {'IP':>7} "
          f"{'W/600PA':>8}{'(nH)':>6} {'W/150IP':>8}{'(nP)':>6}")
    for t in order:
        sub = r[r['tier'] == t]
        if len(sub) < 5:
            continue
        nH = int(sub['war_per_600pa'].notna().sum())
        nP = int(sub['war_per_150ip'].notna().sum())
        print(f"{t:<10} {len(sub):>5} {sub['career_war'].mean():>7.2f} "
              f"{sub['career_g'].mean():>6.0f} {sub['career_pa'].mean():>7.0f} "
              f"{sub['career_ip'].mean():>7.0f} "
              f"{sub['war_per_600pa'].mean():>8.2f}{nH:>6} "
              f"{sub['war_per_150ip'].mean():>8.2f}{nP:>6}")
        val_rows.append({'tier': t, 'n': len(sub),
                         'mean_war': round(sub['career_war'].mean(), 3),
                         'mean_g': round(sub['career_g'].mean(), 1),
                         'mean_pa': round(sub['career_pa'].mean(), 1),
                         'mean_ip': round(sub['career_ip'].mean(), 1),
                         'war_per_600pa': round(sub['war_per_600pa'].mean(), 3),
                         'n_qual_hit': nH,
                         'war_per_150ip': round(sub['war_per_150ip'].mean(), 3),
                         'n_qual_pit': nP})
    pd.DataFrame(val_rows).to_csv('v3_tier_value.csv', index=False)
    print("\nReminder: 2b compares only players who REACHED. Equal here = tiers")
    print("differ on arrival, not on quality/playing-time once there.")
    print("(nH)/(nP) = qualifying hitters (>=145 PA) / pitchers (>=50 IP) per tier.")
    print("A rate cell with a small n in parentheses is noisy -- read with care.")

    # ---- STAGE 2c: models ----
    if HAVE_SM:
        print("\n" + "="*78)
        print("STAGE 2c  models: arrival (logit) + value (OLS), round-controlled")
        print("="*78)
        d = df.copy()
        d['tier'] = pd.Categorical(d['tier'], categories=order, ordered=False)
        try:
            m = smf.logit('reached_mlb ~ C(tier, Treatment("0 Stayer")) + draftRound',
                          data=d).fit(disp=False)
            mrows = []
            print("\nArrival logit (baseline = Stayer):")
            for name in m.params.index:
                if 'tier' in name:
                    orr = np.exp(m.params[name])
                    print(f"  {name:<45} OR={orr:5.2f}  p={m.pvalues[name]:.4f}")
                    mrows.append({'model': 'arrival_logit', 'term': name,
                                  'or': round(orr, 3), 'p': round(m.pvalues[name], 4)})
            rr = r.copy()
            rr['tier'] = pd.Categorical(rr['tier'], categories=order)
            o = smf.ols('career_war ~ C(tier, Treatment("0 Stayer")) + draftRound',
                        data=rr).fit()
            print("\nValue OLS among reachers (career WAR, baseline = Stayer):")
            for name in o.params.index:
                if 'tier' in name:
                    print(f"  {name:<45} coef={o.params[name]:+6.2f}  p={o.pvalues[name]:.4f}")
                    mrows.append({'model': 'value_ols', 'term': name,
                                  'or': round(o.params[name], 3), 'p': round(o.pvalues[name], 4)})
            pd.DataFrame(mrows).to_csv('v3_tier_model.csv', index=False)
        except Exception as e:
            print(f"  model failed: {e}")

    print("\nSaved: v3_tier_breaks / v3_tier_arrival / v3_tier_value / v3_tier_model")
    print("Done.")


if __name__ == '__main__':
    main()
