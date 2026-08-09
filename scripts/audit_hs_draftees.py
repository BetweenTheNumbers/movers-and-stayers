"""
Audit the ORIGINAL HS-draftee mover analysis set (the one already used) for
integrity issues, so we know if the foundation needs cleanup before we build on
it. Checks:

  1. DUPLICATION: is any player counted more than once in the HS analysis?
     (same PlayerID appearing in multiple HS rows within 1996-2019)
  2. LOCATION CONSISTENCY: among HS draftees, any with >1 distinct HS location
     or >1 birth place under one id? (the noise we found for college players)
  3. GEOCODE COVERAGE: what fraction of HS draftees have both birth + HS coords?
     (sanity vs the ~99.6% the study reports)

Works off the analysis file if present, else derives from the register.
Read-only.

Run:  python scripts/audit_hs_draftees.py
"""
import os
import re
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
ANALYSIS = 'v3_analysis_with_war.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    if m:
        return f"{m.group(1).strip()},{m.group(2).strip()}"
    return None


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    # ---- 1 & 2 from the register (authoritative on rows/ids) ----
    reg = pd.read_csv(REG, low_memory=False)
    reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    hs = reg[reg['is_hs']].copy()

    print("="*66)
    print("HS-DRAFTEE ANALYSIS SET AUDIT (1996-2019)")
    print("="*66)
    print(f"  HS-draft rows: {len(hs):,}")
    print(f"  distinct PlayerIDs among them: {hs['PlayerID'].nunique():,}")

    # 1. duplication: players with >1 HS row in-window
    vc = hs['PlayerID'].value_counts()
    dup_ids = vc[vc > 1]
    print(f"\n  players with >1 HS-draft row in 1996-2019: {len(dup_ids):,}")
    print(f"  extra rows from them: {int((dup_ids-1).sum()):,}")
    if len(dup_ids):
        print("  -> if the analysis keeps ALL rows, these are double-counted.")
        print("     (drafted out of HS, didn't sign, drafted from HS again)")

    # 2. location consistency among HS draftees
    hs['hs_loc'] = hs['school'].apply(parse_school_city)
    multi_hs = 0
    multi_birth = 0
    for pid, g in hs.groupby('PlayerID'):
        locs = set(g['hs_loc'].dropna())
        births = set(str(p).strip() for p in g['place'].dropna()
                     if str(p).strip() not in ('', 'nan', '--,--'))
        if len(locs) > 1:
            multi_hs += 1
        if len(births) > 1:
            multi_birth += 1
    print(f"\n  HS draftees with >1 distinct HS location: {multi_hs:,}")
    print(f"  HS draftees with >1 distinct birth place:  {multi_birth:,}")
    if multi_hs or multi_birth:
        print("  -> same label-noise we saw for college; small numbers = pick")
        print("     the HS-draft-row value, low risk.")

    # 3. geocode coverage from the analysis file, if present
    print("\n" + "="*66)
    print("GEOCODE COVERAGE (from analysis file, if available)")
    print("="*66)
    if os.path.exists(ANALYSIS):
        adf = pd.read_csv(ANALYSIS, low_memory=False)
        hsflag = find(adf, 'is_hs_draftee')
        sub = adf[adf[hsflag] == 1] if hsflag else adf
        blat = find(adf, 'birth_lat')
        hlat = find(adf, 'hs_lat')
        dist = find(adf, 'distance_miles', 'dist')
        n = len(sub)
        print(f"  HS draftees in analysis file: {n:,}")
        if blat and hlat:
            both = int((sub[blat].notna() & sub[hlat].notna()).sum())
            print(f"  with birth+HS coords: {both:,} ({both/n*100:.1f}%)")
        if dist:
            hasd = int(sub[dist].notna().sum())
            print(f"  with a computed distance: {hasd:,} ({hasd/n*100:.1f}%)")
        # duplication in the analysis file itself
        apid = find(adf, 'PlayerID', 'playerid')
        if apid:
            advc = sub[apid].value_counts()
            adup = int((advc > 1).sum())
            print(f"  DUPLICATE PlayerIDs in the HS analysis rows: {adup:,}")
            if adup:
                print("  -> the analysis file itself double-counts these.")
    else:
        print(f"  ({ANALYSIS} not found; skipping coverage check)")

    print("\nDone. If duplication ~0 and multi-location small, HS study is clean.")


if __name__ == '__main__':
    main()
