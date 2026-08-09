"""
Exhaustive cross-reference: for the college draftees who currently have NO HS
location, does ANY field in ANY of the 5 source files -- keyed by PlayerID (or
mlbid, or name) -- carry a high-school or hometown we haven't used yet?

For each file: list every location-ish column, and for the GAP players
(no-HS-location college draftees), report how many that column could fill.

Read-only.

Run:  python scripts/crossref_all_files.py
"""
import os
import re
import glob
import pandas as pd

DDIR = 'data'
REG = f'{DDIR}/tbc_draft_register.csv'
SB = f'{DDIR}/tbc_signing_bonus.csv'

LOC_HINTS = ['place', 'city', 'town', 'hs', 'high', 'home', 'origin',
             'hometown', 'birth', 'residence', 'school', 'state', 'geo']


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_place(p):
    p = str(p).strip()
    if ',' in p and p not in ('--,--', 'nan'):
        return True
    return False


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    reg = pd.read_csv(REG, low_memory=False)
    reg['is_hs'] = reg['schoolDivision'].astype(str) == 'HS'
    reg['is_college'] = reg['schoolDivision'].astype(str).isin(
        ['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA', 'NJCAA', 'CCCAA', 'NWAACC'])

    # who already HAS a HS location (backlink OR hsplace)?
    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    backlink_ids = set(hs_rows.loc[hs_rows['loc'].notna(), 'PlayerID'].unique())
    hsplace_ids = set()
    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsp_col = find(sb, 'hsplace')
    if sb_pid and hsp_col:
        for _, r in sb.iterrows():
            if parse_place(r[hsp_col]):
                hsplace_ids.add(r[sb_pid])
    covered = backlink_ids | hsplace_ids

    # the GAP: college 1996-2019 players with NO HS location
    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    col_players = col.drop_duplicates('PlayerID')
    gap_ids = set(col_players['PlayerID']) - covered
    # also gather their mlbids + names for alt-key matching
    gap_rows = col_players[col_players['PlayerID'].isin(gap_ids)]
    gap_mlbids = set(gap_rows['mlbid'].dropna()) - {0}
    gap_names = set(
        (str(a).strip().lower(), str(b).strip().lower())
        for a, b in zip(gap_rows['firstName'], gap_rows['lastName']))

    print(f"GAP players (college, no HS location): {len(gap_ids):,}\n")

    files = sorted(glob.glob(f'{DDIR}/*.csv'))
    print("="*70)
    print("SCANNING EVERY FILE for a field that could fill the gap")
    print("="*70)
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception as e:
            print(f"\n{os.path.basename(f)}: unreadable ({e})")
            continue
        print(f"\n--- {os.path.basename(f)}  ({len(df):,} rows) ---")
        pidc = find(df, 'PlayerID', 'playerid')
        mlbc = find(df, 'mlbid')
        fnc = find(df, 'firstName', 'first')
        lnc = find(df, 'lastName', 'last')

        loc_cols = [c for c in df.columns
                    if any(h in c.lower() for h in LOC_HINTS)]
        if not loc_cols:
            print("  (no location-ish columns)")
            continue

        # how many GAP players even appear in this file, by any key?
        in_by_pid = set()
        if pidc:
            in_by_pid = set(df[pidc]) & gap_ids
        in_by_mlb = set()
        if mlbc:
            in_by_mlb = (set(df[mlbc].dropna()) - {0}) & gap_mlbids
        print(f"  gap players present: by PlayerID {len(in_by_pid):,}"
              + (f" | by mlbid {len(in_by_mlb):,}" if mlbc else ""))

        for c in loc_cols:
            filled = df[c].notna() & (df[c].astype(str).str.strip()
                                      .replace('nan', '').replace('--,--', '') != '')
            # of gap players in this file (by PlayerID), how many have this col filled?
            n_fill = 0
            if pidc:
                sub = df.loc[df[pidc].isin(gap_ids) & filled]
                n_fill = sub[pidc].nunique()
            hs_like = ('hs' in c.lower() or 'high' in c.lower()
                       or 'home' in c.lower() or 'town' in c.lower())
            tag = "  <-- HS/hometown-like!" if hs_like and n_fill > 0 else ""
            print(f"    {c:<20} fills {n_fill:>5,} gap players{tag}")

    print("\n" + "="*70)
    print("READ: any column that 'fills' a large number of gap players AND is")
    print("HS/hometown-like is a NEW route. 'place'/birth columns don't help")
    print("(we already have birth). We need a HIGH-SCHOOL or HOMETOWN field.")


if __name__ == '__main__':
    main()
