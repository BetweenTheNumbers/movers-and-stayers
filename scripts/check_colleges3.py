"""
NEW ROUTE FOUND: the signing-bonus file has a 'colleges3' column = a player's
school history (comma-separated, HS/town first, then colleges). We never used it.
Measure how many of the 10,718 GAP players (college, no HS via backlink/hsplace)
have a colleges3 whose FIRST entry looks like a high school / hometown we could
resolve.

Read-only.

Run:  python scripts/check_colleges3.py
"""
import re
import pandas as pd

REG = 'data/tbc_draft_register.csv'
SB = 'data/tbc_signing_bonus.csv'


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

    sb = pd.read_csv(SB, low_memory=False)
    sb_pid = find(sb, 'playerid', 'PlayerID')
    hsp = find(sb, 'hsplace')
    c3 = find(sb, 'colleges3')
    hsname = find(sb, 'hsName', 'hsname')
    hsplace_ids = set(sb.loc[sb[hsp].apply(parse_place), sb_pid])
    covered = backlink_ids | hsplace_ids

    col = reg.loc[(reg['year'] >= 1996) & (reg['year'] <= 2019) &
                  reg['is_college']].copy()
    players = col.drop_duplicates('PlayerID')
    gap_ids = set(players['PlayerID']) - covered
    print(f"GAP players (no HS via backlink/hsplace): {len(gap_ids):,}\n")

    # signing-file rows for gap players
    sbg = sb[sb[sb_pid].isin(gap_ids)].copy()
    print(f"Gap players present in signing-bonus file: {sbg[sb_pid].nunique():,}")

    def first_token(v):
        s = str(v).strip()
        if s in ('', 'nan'):
            return None
        return s.split(',')[0].strip()

    # colleges3: does it exist, and does first token look like HS/town (not a college)?
    has_c3 = sbg[c3].notna() & (sbg[c3].astype(str).str.strip().replace('nan','') != '')
    print(f"  with a non-empty colleges3:            {int(has_c3.sum()):,}")

    # count how many have a usable HS signal: hsName present, OR colleges3 first
    # token that isn't obviously just the college (heuristic: >=2 tokens)
    n_hsname = 0
    n_c3_multi = 0
    n_any = 0
    examples = []
    for _, r in sbg.iterrows():
        hn = str(r[hsname]).strip() if hsname else ''
        c3v = str(r[c3]).strip() if c3 else ''
        has_hn = hn not in ('', 'nan', '-')
        toks = [t.strip() for t in c3v.split(',')] if c3v not in ('', 'nan') else []
        multi = len(toks) >= 2
        if has_hn:
            n_hsname += 1
        if multi:
            n_c3_multi += 1
        if has_hn or multi:
            n_any += 1
            if len(examples) < 25:
                examples.append((r[sb_pid], r['firstName'], r['lastName'],
                                 hn, c3v))

    print(f"  with hsName:                           {n_hsname:,}")
    print(f"  with colleges3 having >=2 entries:     {n_c3_multi:,}")
    print(f"  with EITHER (a HS signal):             {n_any:,}")
    print(f"\n  >>> potential NEW fills from this file: up to {n_any:,} "
          f"({n_any/len(gap_ids)*100:.1f}% of the gap) <<<\n")

    print("Sample (PID | name | hsName | colleges3):")
    for pid, fn, ln, hn, c3v in examples:
        print(f"  {pid}  {fn} {ln:<18} | hsName: {hn:<28} | c3: {c3v}")


if __name__ == '__main__':
    main()
