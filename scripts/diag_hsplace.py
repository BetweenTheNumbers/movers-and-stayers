"""
DIAGNOSTIC: how usable is hsplace (from signing-bonus) for expanding the mover
analysis to college draftees? Answers, before we build anything:

  1. What fraction of ALL 1996-2019 draftees have an hsplace? Split HS vs college,
     signed vs unsigned -- to size the selection problem.
  2. For the existing HS-draftee analysis, does hsplace agree with the school-
     parsed HS location we already used? (sanity: same variable?)
  3. How many distinct hsplace towns need geocoding beyond the current cache?

Read-only. Nothing is written.

Run:  python scripts/diag_hsplace.py
"""
import os
import sys
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'
CACHE = 'geocode_cache_v3.json'


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    if not (os.path.exists(REG) and os.path.exists(SB)):
        print("ERROR: need register + signing-bonus files.")
        sys.exit(1)
    reg = pd.read_csv(REG, low_memory=False)
    reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
    sb = pd.read_csv(SB, low_memory=False)

    print(f"Register 1996-2019: {len(reg):,} rows")
    print(f"Signing-bonus file:  {len(sb):,} rows\n")

    # keys
    reg_pid = find(reg, 'PlayerID', 'playerid')
    sb_pid = find(sb, 'PlayerID', 'playerid', 'BaseballCubeID', 'bcid')
    hsplace = find(sb, 'hsplace', 'hs_place', 'hstown')
    signed = find(reg, 'signed')
    hsflag = find(reg, 'is_hs_draftee')
    sdiv = find(reg, 'schoolDivision')

    print(f"keys: reg={reg_pid}, sb={sb_pid}, hsplace={hsplace}, signed={signed}")
    if hsplace is None or sb_pid is None:
        print("ERROR: signing-bonus file lacks hsplace or a player id. Columns:")
        print("  " + ", ".join(sb.columns))
        sys.exit(1)

    # which players have a usable (non-empty) hsplace
    sb2 = sb[[sb_pid, hsplace]].copy()
    sb2['has_hs'] = sb2[hsplace].notna() & (sb2[hsplace].astype(str).str.strip()
                                            .replace('nan', '') != '')
    hs_ids = set(sb2[sb2['has_hs']][sb_pid])
    reg['has_hsplace'] = reg[reg_pid].isin(hs_ids)

    # classify college via schoolDivision
    div = reg[sdiv].astype(str)
    reg['is_college'] = div.isin(['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA',
                                  'NJCAA', 'CCCAA', 'NWAACC'])
    reg['is_signed'] = reg[signed].astype(str).str.strip().isin(['Y', '1', 'Yes'])

    print("\n" + "="*66)
    print("hsplace COVERAGE by group (the selection-problem sizing)")
    print("="*66)
    print(f"  {'group':<28}{'n':>8}{'has_hsplace':>13}{'pct':>7}")
    for label, mask in [
        ('HS draftees', reg[hsflag] == 1) if hsflag else ('HS (via !college)', ~reg['is_college']),
        ('College draftees', reg['is_college']),
        ('College + signed', reg['is_college'] & reg['is_signed']),
        ('College + UNsigned', reg['is_college'] & ~reg['is_signed']),
    ]:
        sub = reg[mask]
        n = len(sub)
        h = int(sub['has_hsplace'].sum())
        print(f"  {label:<28}{n:>8}{h:>13}{h/max(n,1)*100:>6.0f}%")

    print("\n  READ: if College+UNsigned has FAR lower hsplace coverage than")
    print("  College+signed, requiring hsplace conditions on signing = selection")
    print("  bias. If coverage is similar across signed/unsigned, we're fine.")

    # distinct hsplace towns to geocode
    print("\n" + "="*66)
    print("GEOCODING NEED for hsplace towns")
    print("="*66)
    sb_join = sb2[sb2['has_hs']].copy()

    def parse_place(p):
        p = str(p).strip()
        if ',' in p:
            c, s = p.rsplit(',', 1)
            return f"{c.strip()}|{s.strip()}"
        return None
    sb_join['key'] = sb_join[hsplace].apply(parse_place)
    towns = set(sb_join['key'].dropna())
    print(f"  distinct hsplace towns: {len(towns):,}")
    if os.path.exists(CACHE):
        import json
        cache = set(json.load(open(CACHE)).keys())
        new = sum(1 for t in towns if t not in cache)
        print(f"  already cached: {len(towns)-new:,} / {len(towns):,}")
        print(f"  NEW to geocode: {new:,}")
    print("\nDone -- diagnostic only.")


if __name__ == '__main__':
    main()
