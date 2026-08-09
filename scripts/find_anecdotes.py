"""
Mine presentation anecdotes from the analysis data (utility, not a step).

Pulls VERIFIED examples straight from the dataset rather than from memory,
so every name on a slide is backed by the same rows the analysis uses.

Categories produced:
  1. Longest moves that reached MLB          (the spectacular cases)
  2. Highest career WAR among movers          (headline movers)
  3. Highest career WAR among stayers         (the honest counterweight —
                                               do not show movers alone)
  4. Players within 1 mile of the cutoff      (threshold-sensitivity slide)
  5. Cold-destination movers                  (moves INTO non-sunbelt states,
                                               which undercut a "they all
                                               moved to Florida" reading)
  6. Foreign-born with a US high school       (the most extreme movers, and
                                               the group your exclusions
                                               analysis is about)
  7. Repeat destination high schools          (do long movers cluster?)

Usage:
    python scripts/find_anecdotes.py
    python scripts/find_anecdotes.py --top 15
    python scripts/find_anecdotes.py --csv
    python scripts/find_anecdotes.py --player "Pujols"

Output:
    anecdotes_<category>.csv   (with --csv)
"""

import os
import sys

import numpy as np
import pandas as pd

ANALYSIS_WAR = 'v3_analysis_with_war.csv'
ANALYSIS = 'v3_analysis.csv'
MOVER_CUTOFF_MI = 5.0

WARM_STATES = {'FL', 'TX', 'CA', 'AZ', 'GA', 'NC', 'SC', 'AL', 'MS', 'LA',
               'NV', 'HI', 'PR'}
DOMESTIC = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI',
            'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI',
            'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC',
            'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
            'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}


def arg(flag, default, cast=int):
    if flag in sys.argv:
        try:
            return cast(sys.argv[sys.argv.index(flag) + 1])
        except (IndexError, ValueError):
            pass
    return default


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def load():
    src = ANALYSIS_WAR if os.path.exists(ANALYSIS_WAR) else ANALYSIS
    if not os.path.exists(src):
        print("ERROR: no analysis CSV found. Run the pipeline first.")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    fn, ln = pick(df, 'firstName'), pick(df, 'lastName')
    df['player'] = (df[fn].fillna('').astype(str) + ' ' +
                    df[ln].fillna('').astype(str)).str.strip()
    df['from'] = df['birth_city'].astype(str) + ', ' + df['birth_state'].astype(str)
    df['to'] = df['hs_city'].astype(str) + ', ' + df['hs_state'].astype(str)
    if 'career_war' not in df.columns:
        df['career_war'] = np.nan
    print(f"Source: {src}  ({len(df):,} rows)")
    return df


def show(title, sub, cols, note=None, csv_name=None):
    print(f"\n{'=' * 96}")
    print(title)
    print("=" * 96)
    if note:
        print(f"  {note}\n")
    if sub.empty:
        print("  (no rows matched)")
        return
    print(sub[cols].to_string(index=False))
    if csv_name and '--csv' in sys.argv:
        sub[cols].to_csv(f'anecdotes_{csv_name}.csv', index=False,
                         encoding='utf-8-sig')
        print(f"\n  saved anecdotes_{csv_name}.csv")


def main():
    top = arg('--top', 12)
    df = load()

    # HS draftees with a computed distance, reached MLB
    hs = df[(df.get('is_hs_draftee', 0) == 1) & df['distance_miles'].notna()].copy()
    mlb = hs[hs['reached_mlb'] == 1].copy()
    print(f"HS draftees with distance: {len(hs):,}   reached MLB: {len(mlb):,}")

    # named-player lookup short-circuits everything else
    name_q = arg('--player', None, str)
    if name_q:
        hit = df[df['player'].str.contains(name_q, case=False, na=False)]
        cols = [c for c in ['player', 'year', 'draftRound', 'from', 'to',
                            'distance_miles', 'mover', 'reached_mlb',
                            'career_war', 'playerClass'] if c in hit.columns]
        show(f'PLAYER LOOKUP: "{name_q}"', hit, cols)
        print()
        return

    base = ['player', 'year', 'draftRound', 'from', 'to', 'distance_miles']
    war = base + ['career_war']

    # 1. longest moves that reached MLB
    show("1. LONGEST MOVES THAT REACHED MLB",
         mlb.nlargest(top, 'distance_miles'), war,
         "The spectacular cases. Check the top row — the sample max is ~9,769 miles.",
         "longest_moves")

    # 2. highest WAR movers
    mv = mlb[mlb['mover'] == 1]
    show("2. HIGHEST CAREER WAR — MOVERS",
         mv.nlargest(top, 'career_war'), war,
         "Headline movers.", "top_movers")

    # 3. highest WAR stayers  (the honest counterweight)
    st = mlb[mlb['mover'] == 0]
    show("3. HIGHEST CAREER WAR — STAYERS",
         st.nlargest(top, 'career_war'), war,
         "Show these ALONGSIDE the movers. Career outcomes are statistically "
         "identical once\n  players reach MLB, so a movers-only slide would "
         "misrepresent the finding.", "top_stayers")

    # 4. players near the threshold
    near = hs[(hs['distance_miles'] >= MOVER_CUTOFF_MI - 1) &
              (hs['distance_miles'] <= MOVER_CUTOFF_MI + 1)]
    near_mlb = near[near['reached_mlb'] == 1]
    show(f"4. WITHIN 1 MILE OF THE {MOVER_CUTOFF_MI:.0f}-MILE CUTOFF (reached MLB)",
         near_mlb.nlargest(top, 'career_war'), war,
         "For the threshold-sensitivity slide: these players flip category on a "
         "cutoff change,\n  and the result holds anyway (OR 1.56-1.69 across "
         "1-100 miles).", "near_threshold")

    # 5. cold-destination movers
    if 'hs_state' in hs.columns:
        cold = mlb[(mlb['mover'] == 1) &
                   (~mlb['hs_state'].astype(str).str.upper().isin(WARM_STATES)) &
                   (mlb['distance_miles'] > 100)]
        show("5. LONG MOVES INTO COLD-WEATHER STATES (reached MLB)",
             cold.nlargest(top, 'career_war'), war,
             "Undercuts a 'they all moved to Florida' reading. Warm-state "
             "coefficients are null\n  (p = 0.59-0.71) — the move matters, not "
             "the destination climate.", "cold_movers")

    # 6. foreign-born with a US high school
    if 'birth_state' in hs.columns:
        foreign = mlb[~mlb['birth_state'].astype(str).str.upper().isin(DOMESTIC)]
        show("6. FOREIGN-BORN WITH A US HIGH SCHOOL (reached MLB)",
             foreign.nlargest(top, 'career_war'), war,
             "The most extreme movers. NOTE: foreign-born players WITHOUT a US "
             "high school are\n  excluded from the sample entirely — that is the "
             "136-player limitation.", "foreign_born")

    # 7. repeat destination high schools for long movers
    hs_name_col = pick(hs, 'hs_name')
    if hs_name_col:
        far = hs[hs['distance_miles'] > 250]
        if len(far):
            g = (far.groupby([hs_name_col, 'to'])
                    .agg(long_movers=('player', 'size'),
                         reached_mlb=('reached_mlb', 'sum'))
                    .reset_index()
                    .sort_values('long_movers', ascending=False))
            g = g[g['long_movers'] >= 3]
            show("7. HIGH SCHOOLS THAT ATTRACT LONG-DISTANCE MOVERS (>250 mi)",
                 g.head(top), [hs_name_col, 'to', 'long_movers', 'reached_mlb'],
                 "Destination clustering. School names are messy in the source, "
                 "so treat as a lead,\n  not a finding.", "destination_schools")

    print(f"\n{'=' * 96}")
    print("Tip: verify any name before it goes on a slide — "
          "python scripts/find_anecdotes.py --player \"Pujols\"")
    print("=" * 96 + "\n")


if __name__ == '__main__':
    main()
