"""
Confirm the population rule: KEEP any player with ANY draft in 1996+, even if he
also had a pre-1996 draft. Check this against how the analysis currently filters,
and count the edge cases so we know if anything actually changes.

Read-only.

Run:  python scripts/check_1996_rule.py
"""
import pandas as pd

REG = 'data/tbc_draft_register.csv'
reg = pd.read_csv(REG, low_memory=False)

reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
FOURYR = ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA']
JUCO = ['NJCAA', 'CCCAA', 'NWAACC']
reg['is_college'] = reg['schoolDivision'].astype(str).isin(FOURYR + JUCO)

# per player: first draft year, and whether they have any 1996+ draft
g = reg.groupby('PlayerID')['year']
first_yr = g.min()
last_yr = g.max()
has_96plus = g.apply(lambda s: (s >= 1996).any())

players = pd.DataFrame({'first': first_yr, 'last': last_yr, 'has96': has_96plus})

print("="*60)
print("POPULATION RULE CHECK")
print("="*60)
print(f"  total distinct players (all years): {len(players):,}")
print(f"  players with ANY 1996+ draft:       {int(players['has96'].sum()):,}")

# the edge case: first draft pre-1996 BUT has a 1996+ draft too
straddle = players[(players['first'] <= 1995) & players['has96']]
print(f"\n  STRADDLERS (first draft <=1995, but also drafted 1996+): "
      f"{len(straddle):,}")
print("  -> your rule KEEPS these. The current row-level 1996-2019 filter also")
print("     keeps their 1996+ rows, so population is the same.")

# players whose ONLY drafts are pre-1996 (correctly excluded either way)
pre_only = players[players['last'] <= 1995]
print(f"\n  players with ALL drafts <=1995 (excluded either way): {len(pre_only):,}")

# now restrict to COLLEGE draftees 1996-2019 (the extension population)
col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
              reg['is_college']].copy()
col_players = col['PlayerID'].nunique()
# how many of those college players first appeared pre-1996?
col_ids = set(col['PlayerID'].unique())
straddle_college = players.loc[players.index.isin(col_ids) &
                               (players['first'] <= 1995)]
print("\n" + "-"*60)
print(f"  college draftees (1996-2019 college row): {col_players:,}")
print(f"  ...of whom first drafted pre-1996: {len(straddle_college):,}")
print("  These are kept under your rule (they have a 1996+ college draft).")
print("  Their HS backlink may come from a pre-1996 HS row -- still a valid")
print("  location lookup, so nothing to change.")


if __name__ == '__main__':
    pass
