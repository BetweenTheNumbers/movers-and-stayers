"""
Step 6 — Join FanGraphs career stats (hitting + pitching) to v3_analysis.csv.

Join key: MLBAMID (FanGraphs) <-> mlbid (draft data, after cleaning)

Inputs:
  v3_analysis.csv
  fg_hit.csv   — FanGraphs career HITTING totals
  fg_pit.csv   — FanGraphs career PITCHING totals
  fg_fld.csv   — FanGraphs career FIELDING totals (cross-check only)

Output:
  v3_analysis_with_war.csv
"""

import sys
import pandas as pd
import numpy as np

DRAFT_FILE = 'v3_analysis.csv'
HIT_FILE   = 'fg_hit.csv'
PIT_FILE   = 'fg_pit.csv'
FLD_FILE   = 'fg_fld.csv'
OUT_FILE   = 'v3_analysis_with_war.csv'


def clean_mlbid(x):
    if pd.isna(x):
        return np.nan
    try:
        s = str(x).strip()
        if s in ('', '0', 'nan', 'None'):
            return np.nan
        return int(float(s))
    except (ValueError, TypeError):
        return np.nan


# 1. Load draftees
draft = pd.read_csv(DRAFT_FILE, low_memory=False)
print(f"Loaded {len(draft)} players from {DRAFT_FILE}")
draft['mlbid_clean'] = draft['mlbid'].apply(clean_mlbid)
n_have = draft['mlbid_clean'].notna().sum()
print(f"  With valid mlbid: {n_have} ({n_have/len(draft)*100:.1f}%)")
print(f"  reached_mlb=1:    {int(draft['reached_mlb'].sum())}")

# 2. Load FanGraphs
print(f"\nLoading FanGraphs files...")
hit = pd.read_csv(HIT_FILE)
pit = pd.read_csv(PIT_FILE)
fld = pd.read_csv(FLD_FILE)

# Guard: confirm each file's CONTENT matches its name. A hitting file must
# have PA (not IP); a pitching file must have IP/ERA. This catches the case
# where fg_hit.csv and fg_pit.csv get swapped on disk, which would otherwise
# silently corrupt career WAR without any error.
def _looks_like(df, kind):
    cols = {c.lower() for c in df.columns}
    if kind == 'hit':
        return 'pa' in cols and 'era' not in cols
    return ('ip' in cols or 'era' in cols) and 'pa' not in cols
if not _looks_like(hit, 'hit'):
    sys.exit(f"ERROR: {HIT_FILE} does not look like a HITTING file "
             f"(expected PA, no ERA). Are fg_hit.csv / fg_pit.csv swapped? "
             f"Run verify_fg_files.py.")
if not _looks_like(pit, 'pit'):
    sys.exit(f"ERROR: {PIT_FILE} does not look like a PITCHING file "
             f"(expected IP/ERA, no PA). Are fg_hit.csv / fg_pit.csv swapped? "
             f"Run verify_fg_files.py.")
print("  file content verified (hit=PA, pit=IP/ERA)")

print(f"  Hitting:  {len(hit)} rows")
print(f"  Pitching: {len(pit)} rows")
print(f"  Fielding: {len(fld)} rows")

for df_fg, name in [(hit, 'hit'), (pit, 'pit'), (fld, 'fld')]:
    df_fg['MLBAMID'] = pd.to_numeric(df_fg['MLBAMID'], errors='coerce')
    df_fg.dropna(subset=['MLBAMID'], inplace=True)
    df_fg['MLBAMID'] = df_fg['MLBAMID'].astype(int)

hit = hit.sort_values('PA', ascending=False).drop_duplicates(subset='MLBAMID', keep='first')
pit = pit.sort_values('IP', ascending=False).drop_duplicates(subset='MLBAMID', keep='first')
fld = fld.sort_values('Inn', ascending=False).drop_duplicates(subset='MLBAMID', keep='first')
print(f"  After dedup -> hit: {len(hit)}, pit: {len(pit)}, fld: {len(fld)}")

# 3. Slim tables
hit_slim = hit.rename(columns={
    'WAR': 'hit_war', 'G': 'hit_games', 'PA': 'hit_pa', 'Off': 'hit_off',
    'Def': 'hit_def', 'BsR': 'hit_bsr', 'wRC+': 'hit_wrc_plus', 'HR': 'hit_hr',
    'AVG': 'hit_avg', 'OBP': 'hit_obp', 'SLG': 'hit_slg', 'wOBA': 'hit_woba',
})[['MLBAMID', 'hit_war', 'hit_games', 'hit_pa', 'hit_off', 'hit_def',
    'hit_bsr', 'hit_wrc_plus', 'hit_hr', 'hit_avg', 'hit_obp', 'hit_slg', 'hit_woba']]

pit_slim = pit.rename(columns={
    'WAR': 'pit_war', 'G': 'pit_games', 'IP': 'pit_ip', 'W': 'pit_w', 'L': 'pit_l',
    'SV': 'pit_sv', 'GS': 'pit_gs', 'ERA': 'pit_era', 'FIP': 'pit_fip',
    'K/9': 'pit_k9', 'BB/9': 'pit_bb9',
})[['MLBAMID', 'pit_war', 'pit_games', 'pit_ip', 'pit_w', 'pit_l',
    'pit_sv', 'pit_gs', 'pit_era', 'pit_fip', 'pit_k9', 'pit_bb9']]

fld_slim = fld[['MLBAMID']].copy()
fld_slim['has_fld_row'] = 1

# 4. Join
print(f"\nJoining FG -> draft on mlbid_clean == MLBAMID...")
enriched = draft.merge(hit_slim, left_on='mlbid_clean', right_on='MLBAMID', how='left').drop(columns=['MLBAMID'])
enriched = enriched.merge(pit_slim, left_on='mlbid_clean', right_on='MLBAMID', how='left').drop(columns=['MLBAMID'])
enriched = enriched.merge(fld_slim, left_on='mlbid_clean', right_on='MLBAMID', how='left').drop(columns=['MLBAMID'])
enriched['has_fld_row'] = enriched['has_fld_row'].fillna(0).astype(int)

# 5. Derived columns
enriched['fg_hit_match'] = enriched['hit_war'].notna().astype(int)
enriched['fg_pit_match'] = enriched['pit_war'].notna().astype(int)
enriched['fg_any_match'] = (enriched['fg_hit_match'] | enriched['fg_pit_match']).astype(int)

enriched['career_war'] = enriched['hit_war'].fillna(0) + enriched['pit_war'].fillna(0)
enriched.loc[enriched['fg_any_match'] == 0, 'career_war'] = np.nan

enriched['is_two_way'] = ((enriched['hit_pa'].fillna(0) >= 50) &
                          (enriched['pit_ip'].fillna(0) >= 20)).astype(int)
enriched['career_games'] = enriched[['hit_games', 'pit_games']].max(axis=1, skipna=True)
enriched['war_per_game'] = enriched['career_war'] / enriched['career_games']

def threshold_flag(war_series, thresh):
    out = (war_series >= thresh).astype('float')
    out[war_series.isna()] = np.nan
    return out

enriched['had_1war_career']  = threshold_flag(enriched['career_war'], 1.0)
enriched['had_3war_career']  = threshold_flag(enriched['career_war'], 3.0)
enriched['had_10war_career'] = threshold_flag(enriched['career_war'], 10.0)
enriched['had_25war_career'] = threshold_flag(enriched['career_war'], 25.0)

def match_status(row):
    if pd.isna(row['mlbid_clean']):
        return 'no_mlbid'
    if row['fg_any_match'] == 0:
        return 'no_fg_match'
    return 'matched'

enriched['mlb_match_status'] = enriched.apply(match_status, axis=1)

# 6. Diagnostics
print(f"\n{'='*70}\nJOIN DIAGNOSTICS\n{'='*70}")
print(f"\nMatch status:\n{enriched['mlb_match_status'].value_counts().to_string()}")
mlb_only = enriched[enriched['reached_mlb'] == 1]
print(f"\nAmong reached_mlb=1 ({len(mlb_only)}):")
print(f"  With FG match: {mlb_only['fg_any_match'].sum()} ({mlb_only['fg_any_match'].mean()*100:.1f}%)")
print(f"  Two-way:       {mlb_only['is_two_way'].sum()}")
print(f"  No FG match:   {(mlb_only['fg_any_match']==0).sum()}")

matched_war = enriched.loc[enriched['fg_any_match'] == 1, 'career_war']
if len(matched_war) > 0:
    print(f"\nCareer WAR distribution (matched, n={len(matched_war)}):")
    print(matched_war.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]).round(2).to_string())

top = enriched.loc[enriched['career_war'].notna()].nlargest(15, 'career_war')
cols = [c for c in ['firstName','lastName','year','draftRound','mlbid_clean',
                    'hit_war','pit_war','career_war','is_two_way'] if c in top.columns]
print(f"\nTop 15 career WAR (sanity check):")
print(top[cols].to_string(index=False))

# 7. Write
enriched.to_csv(OUT_FILE, index=False, encoding='utf-8-sig')
print(f"\nSaved: {OUT_FILE} ({len(enriched)} rows, {len(enriched.columns)} columns)")
