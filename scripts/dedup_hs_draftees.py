"""
Clean the HS-draftee analysis set:
  - DEDUP the 54 players with >1 HS-draft row: keep the SIGNED row; if none
    signed, keep the LATEST draft year.
  - For the ~11 with conflicting HS locations, the kept row's HS location wins
    automatically (same rule), fixing wrong distances.

Works on the register to decide which (PlayerID, year) row to keep, then filters
the analysis file to those kept rows. Reports before/after n and the headline
mover/stayer rates so you can confirm the change is tiny (correctness check).

Writes: v3_analysis_hs_deduped.csv  (HS draftees only, one row per player)

Run:  python scripts/dedup_hs_draftees.py
"""
import os
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
ANALYSIS = 'v3_analysis_with_war.csv'
OUT = 'v3_analysis_hs_deduped.csv'


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    if not (os.path.exists(REG) and os.path.exists(ANALYSIS)):
        print("ERROR: need register + analysis file.")
        sys.exit(1)

    reg = pd.read_csv(REG, low_memory=False)
    reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    hs = reg[reg['is_hs']].copy()
    hs['signed_flag'] = hs['signed'].astype(str).str.strip().isin(
        ['Y', '1', 'Yes']).astype(int)

    # ---- choose ONE row per player: signed first, then latest year ----
    # sort so the preferred row is last, then keep last per PlayerID
    hs_sorted = hs.sort_values(['PlayerID', 'signed_flag', 'year'])
    kept = hs_sorted.drop_duplicates('PlayerID', keep='last').copy()

    n_before = len(hs)
    n_after = len(kept)
    print("="*62)
    print("HS DEDUP (keep signed row, else latest)")
    print("="*62)
    print(f"  HS-draft rows before: {n_before:,}")
    print(f"  distinct players:     {hs['PlayerID'].nunique():,}")
    print(f"  rows kept (1/player): {n_after:,}")
    print(f"  rows dropped:         {n_before - n_after:,}")

    # kept (PlayerID, year) pairs define which analysis rows survive
    keep_keys = set(zip(kept['PlayerID'], kept['year']))

    # ---- filter the analysis file to kept rows ----
    adf = pd.read_csv(ANALYSIS, low_memory=False)
    hsflag = find(adf, 'is_hs_draftee')
    apid = find(adf, 'PlayerID', 'playerid')
    ayear = find(adf, 'year')
    reached = find(adf, 'reached_mlb')
    mover = find(adf, 'mover')
    dist = find(adf, 'distance_miles', 'dist')

    hs_an = adf[adf[hsflag] == 1].copy() if hsflag else adf.copy()

    # headline BEFORE
    def rates(df):
        if not (reached and mover):
            return None
        m = df[df[mover] == 1]
        s = df[df[mover] == 0]
        return (m[reached].mean()*100 if len(m) else float('nan'),
                s[reached].mean()*100 if len(s) else float('nan'),
                len(m), len(s))

    before = rates(hs_an)

    # keep only rows whose (PlayerID, year) is in keep_keys; if analysis file
    # has no year, dedup by PlayerID keeping first (rare)
    if ayear:
        mask = [ (p, y) in keep_keys for p, y in zip(hs_an[apid], hs_an[ayear]) ]
        hs_clean = hs_an[pd.Series(mask, index=hs_an.index)].copy()
        # any player still duplicated (analysis year mismatch)? keep first
        hs_clean = hs_clean.drop_duplicates(apid, keep='first')
    else:
        hs_clean = hs_an.drop_duplicates(apid, keep='first').copy()

    after = rates(hs_clean)

    print(f"\n  analysis HS rows before: {len(hs_an):,}")
    print(f"  analysis HS rows after:  {len(hs_clean):,}")
    print(f"  removed duplicates:      {len(hs_an) - len(hs_clean):,}")

    if before and after:
        print("\n  HEADLINE mover/stayer reach rate (correctness check):")
        print(f"    BEFORE: movers {before[0]:.2f}% (n={before[2]:,}) | "
              f"stayers {before[1]:.2f}% (n={before[3]:,})")
        print(f"    AFTER:  movers {after[0]:.2f}% (n={after[2]:,}) | "
              f"stayers {after[1]:.2f}% (n={after[3]:,})")
        print("    (should be nearly identical -- confirms dedup is safe)")

    hs_clean.to_csv(OUT, index=False)
    print(f"\nSaved deduped HS analysis: {OUT} ({len(hs_clean):,} rows)")


if __name__ == '__main__':
    main()
