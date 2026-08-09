"""
Pathway split of HS draftees: does the mover/stayer -> MLB effect hold across
different post-HS-draft pathways?

For each HS draftee (1996-2019) derive:
  - signed_from_hs   : signed from their HS-draft row (entered pro from HS)
  - hs_was_last      : HS draft was their FINAL draft appearance (never redrafted)
  - n_redrafts       : # of drafts AFTER the HS draft (0 = none; college redrafts)
Then show mover vs stayer reach-MLB rates WITHIN each pathway group, side by side.

Uses the register for pathway (all a player's draft rows) and the analysis file
for mover flag + reached_mlb + distance.

Read-only (writes hs_pathway_split.csv summary).

Run:  python scripts/hs_pathway_split.py
"""
import os
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
ANALYSIS = 'v3_analysis_with_war.csv'


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
    reg['signed_flag'] = reg['signed'].astype(str).str.strip().isin(
        ['Y', '1', 'Yes']).astype(int)
    reg['is_hs_row'] = reg['schoolDivision'].astype(str) == 'HS'

    # ---- per-player pathway from ALL their draft rows ----
    path = {}
    for pid, g in reg.groupby('PlayerID'):
        g = g.sort_values('year')
        hs = g[g['is_hs_row']]
        if len(hs) == 0:
            continue  # not a HS draftee at all
        hs_year = int(hs['year'].min())            # first HS draft
        signed_hs = int(hs.loc[hs['year'] == hs_year, 'signed_flag'].max())
        after = g[g['year'] > hs_year]             # drafts after HS draft
        n_redrafts = int(len(after))
        hs_was_last = int(int(g['year'].max()) == hs_year)
        path[pid] = dict(signed_from_hs=signed_hs, n_redrafts=n_redrafts,
                         hs_was_last=hs_was_last)
    pathdf = pd.DataFrame.from_dict(path, orient='index')

    # ---- join to analysis (HS draftees, mover flag, reached_mlb) ----
    adf = pd.read_csv(ANALYSIS, low_memory=False)
    hsflag = find(adf, 'is_hs_draftee')
    apid = find(adf, 'PlayerID', 'playerid')
    reached = find(adf, 'reached_mlb')
    mover = find(adf, 'mover')
    hs = adf[adf[hsflag] == 1].copy()
    hs = hs.merge(pathdf, left_on=apid, right_index=True, how='left')

    n = len(hs)
    print("="*66)
    print(f"HS DRAFTEES: {n:,}   (pathway derived from register)")
    print("="*66)
    print(f"  signed from HS:        {int((hs['signed_from_hs']==1).sum()):,} "
          f"({(hs['signed_from_hs']==1).mean()*100:.1f}%)")
    print(f"  did NOT sign from HS:  {int((hs['signed_from_hs']==0).sum()):,}")
    print(f"  HS draft was last:     {int((hs['hs_was_last']==1).sum()):,} "
          f"({(hs['hs_was_last']==1).mean()*100:.1f}%)")
    print(f"  redrafted >=1 time:    {int((hs['n_redrafts']>=1).sum()):,}")
    print(f"  redraft count distribution: "
          f"{hs['n_redrafts'].value_counts().sort_index().to_dict()}")

    def split(df, label):
        m = df[df[mover] == 1]
        s = df[df[mover] == 0]
        mr = m[reached].mean()*100 if len(m) else float('nan')
        sr = s[reached].mean()*100 if len(s) else float('nan')
        lift = mr - sr
        # crude OR
        try:
            a = m[reached].sum(); b = len(m)-a
            c = s[reached].sum(); d = len(s)-c
            orr = (a*d)/(b*c) if b and c else float('nan')
        except Exception:
            orr = float('nan')
        print(f"  {label:<26} movers {mr:5.1f}% (n={len(m):>5,}) | "
              f"stayers {sr:5.1f}% (n={len(s):>5,}) | "
              f"lift {lift:+5.1f}pp | OR {orr:4.2f}")
        return dict(group=label, mover_rate=mr, stayer_rate=sr, lift=lift,
                    OR=orr, n_mover=len(m), n_stayer=len(s))

    print("\n" + "="*66)
    print("MOVER vs STAYER -> MLB, by pathway (side by side)")
    print("="*66)
    rows = []
    rows.append(split(hs, "ALL HS draftees"))
    print()
    rows.append(split(hs[hs['signed_from_hs'] == 1], "Signed from HS"))
    rows.append(split(hs[hs['signed_from_hs'] == 0], "Did NOT sign from HS"))
    print()
    rows.append(split(hs[hs['hs_was_last'] == 1], "HS draft was last"))
    rows.append(split(hs[hs['hs_was_last'] == 0], "Redrafted later"))
    print()
    rows.append(split(hs[hs['n_redrafts'] == 0], "0 redrafts"))
    rows.append(split(hs[hs['n_redrafts'] == 1], "1 redraft"))
    rows.append(split(hs[hs['n_redrafts'] >= 2], "2+ redrafts"))

    pd.DataFrame(rows).to_csv('hs_pathway_split.csv', index=False)
    print("\nSaved hs_pathway_split.csv")
    print("\nNOTE: for redrafted players, 'reached MLB' came via a LATER (college)")
    print("draft, so their HS-era mover status -> MLB is a robustness signal, not")
    print("a clean pathway. Signed-from-HS is the cleanest test.")


if __name__ == '__main__':
    main()
