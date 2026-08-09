"""
For a sample of the gap players (college, no HS location), dump their ACTUAL row
values from EVERY The Baseball Cube file -- every column, not just location-ish ones --
so we can see if any file holds a hometown/address/region/HS field with real
content we overlooked. Matches on PlayerID and mlbid.

Read-only.

Run:  python scripts/dump_gap_rows.py
"""
import re
import glob
import os
import pandas as pd

DDIR = 'data'
REG = f'{DDIR}/tbc_draft_register.csv'
SB = f'{DDIR}/tbc_signing_bonus.csv'


def parse_school_city(s):
    m = re.search(r'\(([^,]+),\s*([A-Za-z]{2})\)\s*$', str(s))
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def parse_place(p):
    p = str(p).strip()
    return ',' in p and p not in ('--,--', 'nan')


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
        ['NCAA 1','NCAA 2','NCAA 3','NAIA','NJCAA','CCCAA','NWAACC'])

    hs_rows = reg.loc[reg['is_hs']].copy()
    hs_rows['loc'] = hs_rows['school'].apply(parse_school_city)
    backlink_ids = set(hs_rows.loc[hs_rows['loc'].notna(), 'PlayerID'].unique())
    sb0 = pd.read_csv(SB, low_memory=False)
    sb_pid0 = find(sb0, 'playerid', 'PlayerID')
    hsp0 = find(sb0, 'hsplace')
    hsplace_ids = set(sb0.loc[sb0[hsp0].apply(parse_place), sb_pid0])
    covered = backlink_ids | hsplace_ids

    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy().sort_values('overall')
    players = col.drop_duplicates('PlayerID', keep='first')
    gap = players[~players['PlayerID'].isin(covered)]

    # sample: the 8 highest picks in the gap (recognizable, easy to sanity-check)
    sample = gap.sort_values('overall').head(8)
    sample_pids = list(sample['PlayerID'])
    sample_mlbids = [m for m in sample['mlbid'].tolist() if m and m != 0]

    print("SAMPLE gap players (highest picks):")
    for _, r in sample.iterrows():
        print(f"  PID {r['PlayerID']}  mlbid {r['mlbid']}  "
              f"{r['firstName']} {r['lastName']}  (#{int(r['overall'])}, "
              f"{int(r['year'])})  born {r['place']}")
    print()

    files = sorted(glob.glob(f'{DDIR}/*.csv'))
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        pidc = find(df, 'PlayerID', 'playerid')
        mlbc = find(df, 'mlbid')
        # rows matching our sample by either key
        m = pd.Series(False, index=df.index)
        if pidc:
            m = m | df[pidc].isin(sample_pids)
        if mlbc:
            m = m | df[mlbc].isin(sample_mlbids)
        hits = df[m]
        print("="*74)
        print(f"{os.path.basename(f)}  ({len(df):,} rows) \u2014 {len(hits)} sample matches")
        print("="*74)
        if len(hits) == 0:
            print("  (none of the sample players appear here)\n")
            continue
        # print every NON-empty column value for each hit
        for _, r in hits.iterrows():
            who = ''
            fnc = find(df, 'firstName', 'first'); lnc = find(df, 'lastName', 'last')
            if fnc and lnc:
                who = f"{r[fnc]} {r[lnc]}"
            print(f"  --- {who}  (PID {r[pidc] if pidc else '?'}) ---")
            for c in df.columns:
                v = r[c]
                sv = str(v).strip()
                if sv not in ('', 'nan', '--,--', '-', '0'):
                    print(f"      {c:<18}: {sv[:50]}")
            print()


if __name__ == '__main__':
    main()
