"""
Step 1 — Build the mover/stayer base dataset from the The Baseball Cube data.

Outputs:
  v3_analysis.csv — all draftees in the configured window (see scripts/config.py)
  with parsed locations and outcome flags.
"""

import pandas as pd
import numpy as np
import re
import os
from config import START_YEAR, END_YEAR, COHORT_LABEL

SRC = 'data/tbc_draft_register.csv'

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} total rows")

# === 1. Filter to the configured draft-year window ===
df = df[(df['year'] >= START_YEAR) & (df['year'] <= END_YEAR)].copy()
print(f"{COHORT_LABEL} rows: {len(df)}")

# === 2. Parse 'place' into birth_city, birth_state ===
def parse_place(p):
    if pd.isna(p):
        return pd.Series([None, None])
    p = str(p).strip()
    if ',' in p:
        parts = [x.strip() for x in p.rsplit(',', 1)]
        return pd.Series([parts[0], parts[1]])
    return pd.Series([p, None])

df[['birth_city', 'birth_state']] = df['place'].apply(parse_place)
print(f"Birth city parsed: {df['birth_city'].notna().sum()} ({df['birth_city'].notna().mean()*100:.1f}%)")
print(f"Birth state parsed: {df['birth_state'].notna().sum()}")

# === 3. Parse 'school' into school_name, school_city, school_state ===
# Format: "Glynn Academy (Brunswick,GA)" or "Wittenberg (Wittenberg,WI)"
def parse_school(s):
    if pd.isna(s):
        return pd.Series([None, None, None])
    s = str(s).strip()
    m = re.search(r'\(([^()]+)\)\s*$', s)
    if not m:
        return pd.Series([s, None, None])
    inside = m.group(1).strip()
    school_name = s[:m.start()].strip()
    if ',' in inside:
        parts = [x.strip() for x in inside.rsplit(',', 1)]
        return pd.Series([school_name, parts[0], parts[1]])
    return pd.Series([school_name, inside, None])

df[['school_name', 'school_city', 'school_state']] = df['school'].apply(parse_school)
print(f"School name parsed: {df['school_name'].notna().sum()}")
print(f"School city parsed: {df['school_city'].notna().sum()} ({df['school_city'].notna().mean()*100:.1f}%)")
print(f"School state parsed: {df['school_state'].notna().sum()}")

# === 4. Outcome variable: reached MLB ===
df['reached_mlb'] = (df['highLevel'] == 'MLB').astype(int)

# === 5. Clean signed flag ===
df['signed_flag'] = (df['signed'].astype(str).str.strip() == 'Y').astype(int)

# === 6. Identify HS draftees ===
df['is_hs_draftee'] = ((df['playerClass'] == 'HS') | (df['schoolDivision'] == 'HS')).astype(int)
print(f"\nHS draftees: {df['is_hs_draftee'].sum()}")
print(f"Non-HS draftees: {(df['is_hs_draftee']==0).sum()}")

# === 7. Coverage report ===
print(f"\n{'='*60}")
print("OVERALL COVERAGE")
print(f"{'='*60}")
print(f"Total rows ({COHORT_LABEL}):              {len(df)}")
print(f"  Has birth city & state:           {(df['birth_city'].notna() & df['birth_state'].notna()).sum()}")
print(f"  Has school city & state:          {(df['school_city'].notna() & df['school_state'].notna()).sum()}")
print(f"  Has BOTH locations parsed:        {(df['birth_city'].notna() & df['birth_state'].notna() & df['school_city'].notna() & df['school_state'].notna()).sum()}")

print(f"\n--- HS Draftees Coverage ---")
hs = df[df['is_hs_draftee']==1]
print(f"  Total HS draftees:                {len(hs)}")
print(f"  Birth city present:               {hs['birth_city'].notna().sum()} ({hs['birth_city'].notna().mean()*100:.1f}%)")
print(f"  School city present:              {hs['school_city'].notna().sum()} ({hs['school_city'].notna().mean()*100:.1f}%)")
both_hs = (hs['birth_city'].notna() & hs['school_city'].notna())
print(f"  Has both:                         {both_hs.sum()} ({both_hs.mean()*100:.1f}%)")
print(f"  Reached MLB:                      {hs['reached_mlb'].sum()} ({hs['reached_mlb'].mean()*100:.1f}%)")

# === 8. Quick same-city vs different-city check (no geocoding yet) ===
print(f"\n{'='*60}")
print("QUICK MOVER/STAYER CHECK (city name comparison only)")
print(f"{'='*60}")

hs_full = hs[hs['birth_city'].notna() & hs['school_city'].notna()].copy()
hs_full['same_city'] = hs_full['birth_city'].str.lower().str.strip() == hs_full['school_city'].str.lower().str.strip()

print(f"\nHS draftees with both locations: {len(hs_full)}")
print(f"  Same city (stayer):     {hs_full['same_city'].sum()} ({hs_full['same_city'].mean()*100:.1f}%)")
print(f"  Different city (mover):  {(~hs_full['same_city']).sum()} ({(~hs_full['same_city']).mean()*100:.1f}%)")

print(f"\nMLB reach rates:")
stayers = hs_full[hs_full['same_city']]
movers = hs_full[~hs_full['same_city']]
print(f"  Stayers: {len(stayers):5d} drafted, {stayers['reached_mlb'].sum():4d} MLB ({stayers['reached_mlb'].mean()*100:.1f}%)")
print(f"  Movers:  {len(movers):5d} drafted, {movers['reached_mlb'].sum():4d} MLB ({movers['reached_mlb'].mean()*100:.1f}%)")
print(f"  Difference: {(movers['reached_mlb'].mean()-stayers['reached_mlb'].mean())*100:+.1f} pp")

from scipy import stats
ct = pd.crosstab(hs_full['same_city'], hs_full['reached_mlb'])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"  Chi-squared: {chi2:.2f}, p = {p:.8f}")

print(f"\nSplit by signed status:")
for s, label in [(1, 'Signed'), (0, 'Did Not Sign')]:
    sub = hs_full[hs_full['signed_flag']==s]
    if len(sub) > 50:
        mv = sub[~sub['same_city']]
        st = sub[sub['same_city']]
        print(f"  {label}: total {len(sub)}, movers {mv['reached_mlb'].mean()*100:.1f}% (n={len(mv)}), stayers {st['reached_mlb'].mean()*100:.1f}% (n={len(st)})")

df.to_csv('v3_analysis.csv', index=False, encoding='utf-8-sig')
print(f"\nSaved full dataset: v3_analysis.csv ({len(df)} rows)")
