"""
Is the bat/arm mispricing CHANGING OVER TIME?

The coverage-gap analysis found hitters reach MLB LESS than their draft slot
predicts while pitchers reach MORE -- in every region. Question: is this edge
still live in recent drafts, or has it been arbitraged away as hitting
projection improved? Recent history is what matters for a front office.

Measured the same way for every era via a FIXED draft+6 window (draft year
through draft+6 inclusive, so every cohort is on the same 7-season clock and
recent drafts are not penalized for not-yet-debuting). Reach-in-window is built
from the CAPPED (<=2025) season files, not the all-time static field.

For each era bucket AND each draft year:
  hitter_resid  = mean(reached_in_window - round_expected)  for hitters
  pitcher_resid = same for pitchers
  gap           = pitcher_resid - hitter_resid   (how much arms beat bats vs slot)
A shrinking gap over time => the edge is closing. A stable gap => still live.

Round-expectation is fit WITHIN each era (a 20th-round pick's baseline reach
changed over time), so the residual is always relative to that era's own draft.

Run:  python scripts/v3_batarm_trend.py
Outputs: v3_batarm_era.csv, v3_batarm_year.csv
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

MAX_SEASON_CAP = 2025
WINDOW_SPAN = 7          # draft year + 6, inclusive
ERAS = [('1996-2003', 1996, 2003), ('2004-2011', 2004, 2011),
        ('2012-2019', 2012, 2019)]
PITCHER_POS = {'P', 'RHP', 'LHP', 'PITCHER', 'SP', 'RP'}
HIT_SEASONS = 'fg-hit-seasons.csv'
PIT_SEASONS = 'fg-pit-seasons.csv'


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


def detect_kind(cols):
    cl = set(c.upper() for c in cols)
    if 'ERA' in cl or 'IP' in cl:
        return 'PIT'
    if 'PA' in cl or 'WOBA' in cl or 'WRC+' in cl:
        return 'HIT'
    return 'HIT'


def load():
    src = 'v3_analysis_with_war.csv' if os.path.exists('v3_analysis_with_war.csv') \
        else 'v3_analysis.csv'
    df = pd.read_csv(src, low_memory=False)
    if 'is_hs_draftee' in df.columns:
        df = df[df['is_hs_draftee'] == 1]
    df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()
    # type
    poscol = next((c for c in df.columns if c.lower() in
                   ('draftposit', 'pos', 'position', 'draft_pos')), None)
    df['dtype'] = df[poscol].apply(
        lambda p: 'Pitcher' if is_pitcher_pos(p) is True
        else ('Hitter' if is_pitcher_pos(p) is False else None)) if poscol else None
    return df


def windowed_reach(df):
    """reached within draft..draft+6 from capped season files -> set of ids."""
    if not (os.path.exists(HIT_SEASONS) and os.path.exists(PIT_SEASONS)):
        print("ERROR: season files required.")
        sys.exit(1)
    a = pd.read_csv(HIT_SEASONS, low_memory=False)
    b = pd.read_csv(PIT_SEASONS, low_memory=False)
    hit = a if detect_kind(a.columns) == 'HIT' else b
    pit = b if detect_kind(b.columns) == 'PIT' else a
    idc = 'MLBAMID' if 'MLBAMID' in hit.columns else 'PlayerId'
    frames = []
    for d in (hit, pit):
        d['Season'] = pd.to_numeric(d['Season'], errors='coerce')
        frames.append(d[d['Season'] <= MAX_SEASON_CAP][[idc, 'Season']])
    seas = pd.concat(frames, ignore_index=True).dropna()
    acol = next((c for c in ['mlbid', 'MLBAMID', 'mlbid_clean'] if c in df.columns), None)
    df['_id'] = pd.to_numeric(df[acol], errors='coerce')
    dy = df.dropna(subset=['_id']).set_index('_id')['year'].to_dict()
    seas['dy'] = seas[idc].map(dy)
    seas = seas.dropna(subset=['dy'])
    inwin = seas[(seas['Season'] >= seas['dy']) &
                 (seas['Season'] <= seas['dy'] + WINDOW_SPAN - 1)]
    reached_ids = set(inwin[idc].unique())
    df['reached_win'] = df['_id'].isin(reached_ids).astype(int)
    return df


def resid_by_type(sub):
    """round-expected reach fit within sub; return per-type mean residual."""
    rcol = 'draftRound' if 'draftRound' in sub.columns else 'round'
    s = sub.dropna(subset=['dtype']).copy()
    if len(s) < 50 or s['reached_win'].sum() < 5:
        return None
    if HAVE_SM:
        try:
            m = smf.logit(f'reached_win ~ {rcol}', data=s).fit(disp=False)
            s['p'] = m.predict(s)
        except Exception:
            s['p'] = s.groupby(rcol)['reached_win'].transform('mean')
    else:
        s['p'] = s.groupby(rcol)['reached_win'].transform('mean')
    s['resid'] = s['reached_win'] - s['p']
    h = s[s['dtype'] == 'Hitter']['resid'].mean()
    p = s[s['dtype'] == 'Pitcher']['resid'].mean()
    nh = int((s['dtype'] == 'Hitter').sum())
    npi = int((s['dtype'] == 'Pitcher').sum())
    return h, p, nh, npi


def main():
    df = load()
    print(f"Rows {COHORT_LABEL}: {len(df):,}   type coverage: "
          f"{df['dtype'].notna().mean()*100:.0f}%")
    df = windowed_reach(df)
    print(f"Reached within draft+6 (<= {MAX_SEASON_CAP}): "
          f"{int(df['reached_win'].sum()):,} "
          f"({df['reached_win'].mean()*100:.1f}%)\n")

    # ---- ERA buckets ----
    print("="*72)
    print("ERA buckets -- reach residual vs round-expected, by type (draft+6)")
    print("gap = pitcher_resid - hitter_resid  (how much arms beat bats vs slot)")
    print("="*72)
    print(f"  {'era':<12}{'nH':>6}{'nP':>6}{'hit_res':>9}{'pit_res':>9}{'gap':>8}")
    erows = []
    for label, y0, y1 in ERAS:
        sub = df[(df['year'] >= y0) & (df['year'] <= y1)]
        out = resid_by_type(sub)
        if out is None:
            continue
        h, p, nh, npi = out
        print(f"  {label:<12}{nh:>6}{npi:>6}{h:>+9.3f}{p:>+9.3f}{p-h:>+8.3f}")
        erows.append({'era': label, 'n_hit': nh, 'n_pit': npi,
                      'hit_resid': round(h, 4), 'pit_resid': round(p, 4),
                      'gap': round(p-h, 4)})
    pd.DataFrame(erows).to_csv('v3_batarm_era.csv', index=False)

    # ---- yearly trend ----
    print("\n" + "="*72)
    print("YEARLY trend -- bat/arm gap by draft year (noisier; watch the shape)")
    print("="*72)
    print(f"  {'year':<6}{'nH':>5}{'nP':>5}{'hit_res':>9}{'pit_res':>9}{'gap':>8}")
    yrows = []
    for yr in range(START_YEAR, END_YEAR+1):
        sub = df[df['year'] == yr]
        out = resid_by_type(sub)
        if out is None:
            continue
        h, p, nh, npi = out
        print(f"  {yr:<6}{nh:>5}{npi:>5}{h:>+9.3f}{p:>+9.3f}{p-h:>+8.3f}")
        yrows.append({'year': yr, 'n_hit': nh, 'n_pit': npi,
                      'hit_resid': round(h, 4), 'pit_resid': round(p, 4),
                      'gap': round(p-h, 4)})
    ydf = pd.DataFrame(yrows)
    ydf.to_csv('v3_batarm_year.csv', index=False)

    # simple trend read: OLS of gap on year
    if HAVE_SM and len(ydf) > 4:
        m = smf.ols('gap ~ year', data=ydf).fit()
        slope = m.params['year']
        print(f"\n  gap-vs-year slope: {slope:+.4f}/yr  p={m.pvalues['year']:.3f}")
        if m.pvalues['year'] < 0.10:
            direction = "SHRINKING (edge closing)" if slope < 0 else "WIDENING"
            print(f"  -> bat/arm gap is {direction} over time.")
        else:
            print(f"  -> no significant time trend; gap looks STABLE.")

    print("\nSaved: v3_batarm_era.csv, v3_batarm_year.csv")
    print("READ: if the gap is stable/large in 2012-2019, the mispricing is still")
    print("live and actionable. If it shrinks to ~0 recently, the edge has closed.")


if __name__ == '__main__':
    main()
