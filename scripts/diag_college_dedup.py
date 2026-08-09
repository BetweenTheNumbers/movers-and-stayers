"""
DIAGNOSTIC ONLY -- reports the situation before we rebuild for college inclusion.
Changes nothing. Answers three questions:

  1. How many players appear in the register MORE THAN ONCE (multi-drafted)?
     These are the double-count risk when we add college draftees. Break down by
     signed status so we can pick a dedup rule.
  2. What's the HS vs college split, and how many of each have a usable
     birth + school location (school = HS for HS guys, college for college guys)?
  3. For the "both distances" idea: is there ANY separate HS location stored for
     college draftees? (Expected: no -- the 'school' field holds whichever school
     he was drafted from.)

Run:  python scripts/diag_college_dedup.py
"""

import os
import sys
import pandas as pd

REGISTER = 'data/tbc_draft_register.csv'
ANALYSIS = 'v3_analysis_with_war.csv'


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    src = ANALYSIS if os.path.exists(ANALYSIS) else \
        ('v3_analysis.csv' if os.path.exists('v3_analysis.csv') else REGISTER)
    if not os.path.exists(src):
        print(f"ERROR: none of the expected files found (looked for {src}).")
        sys.exit(1)
    df = pd.read_csv(src, low_memory=False)
    print(f"Loaded {src}: {len(df):,} rows, {len(df.columns)} cols\n")

    pid = find(df, 'mlbid', 'PlayerID', 'retroid', 'playerid')
    yr = find(df, 'year')
    signed = find(df, 'signed_flag', 'signed')
    hsflag = find(df, 'is_hs_draftee')
    pclass = find(df, 'playerClass')
    sdiv = find(df, 'schoolDivision')
    bcity = find(df, 'birth_city')
    bstate = find(df, 'birth_state')
    scity = find(df, 'school_city')
    sstate = find(df, 'school_state')

    # ---- 1. multi-draft players ----
    print("="*70)
    print("1. MULTI-DRAFT PLAYERS (double-count risk)")
    print("="*70)
    if pid is None:
        print("  No player-id column found; cannot assess multi-draft. Cols:")
        print("  " + ", ".join(df.columns[:30]))
    else:
        idv = df[df[pid].notna() & (df[pid].astype(str).str.strip() != '')]
        vc = idv[pid].value_counts()
        multi = vc[vc > 1]
        print(f"  Player-id column: {pid}")
        print(f"  Rows with an id: {len(idv):,}   distinct players: {idv[pid].nunique():,}")
        print(f"  Players appearing >1 time: {len(multi):,} "
              f"({len(multi)/max(idv[pid].nunique(),1)*100:.1f}% of id'd players)")
        print(f"  Extra rows they create (double-count size): "
              f"{int((multi-1).sum()):,}")
        if hsflag and len(multi) > 0:
            # of the multi-drafted, how many have both a HS and a college row?
            sub = idv[idv[pid].isin(multi.index)]
            mix = sub.groupby(pid)[hsflag].nunique()
            both = int((mix > 1).sum())
            print(f"  Of those, drafted as BOTH HS and college at different times: "
                  f"{both:,}")
        if signed:
            sv = df[signed].astype(str).str.strip()
            ssigned = sv.isin(['1', 'Y', 'Yes', 'True'])
            print(f"  Signed rows: {int(ssigned.sum()):,} / {len(df):,}")

    # ---- 2. HS vs college split + location coverage ----
    print("\n" + "="*70)
    print("2. HS vs COLLEGE SPLIT + birth/school location coverage")
    print("="*70)
    if hsflag:
        for val, label in [(1, 'HS draftee'), (0, 'College/other')]:
            sub = df[df[hsflag] == val]
            has_loc = 0
            if bcity and scity:
                has_loc = int((sub[bcity].notna() & sub[scity].notna()).sum())
            print(f"  {label:<16}: {len(sub):>7,}   with birth+school location: "
                  f"{has_loc:>7,} ({has_loc/max(len(sub),1)*100:.0f}%)")
    else:
        print("  No is_hs_draftee column here (may be register, not analysis file).")
        if pclass:
            print(f"  playerClass values: "
                  f"{df[pclass].value_counts().head(8).to_dict()}")
        if sdiv:
            print(f"  schoolDivision values: "
                  f"{df[sdiv].value_counts().head(8).to_dict()}")

    # ---- 3. is there a separate HS location for college draftees? ----
    print("\n" + "="*70)
    print("3. SEPARATE HS LOCATION FOR COLLEGE DRAFTEES? (for HS->college distance)")
    print("="*70)
    hs_loc_cols = [c for c in df.columns if 'hs_' in c.lower() or 'highschool' in c.lower()
                   or 'high_school' in c.lower()]
    col_loc_cols = [c for c in df.columns if 'college' in c.lower()
                    or 'univ' in c.lower()]
    print(f"  Columns mentioning HS: {hs_loc_cols or 'NONE'}")
    print(f"  Columns mentioning college: {col_loc_cols or 'NONE'}")
    print("  (If both are NONE, the 'school' field holds only the draft-time school,")
    print("   so HS->college distance is NOT computable from this data -- college")
    print("   movers would be measured birth->college instead.)")

    print("\nDone -- nothing changed. Use these numbers to choose the dedup rule.")


if __name__ == '__main__':
    main()
