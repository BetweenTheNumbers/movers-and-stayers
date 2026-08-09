"""
For each flagged/ambiguous college in the crosswalk, pull a few SAMPLE PLAYERS
(name, year, round, overall, birth place, division) so they can be looked up
manually (Baseball Reference, FanGraphs draft pages) to confirm which school
the register actually means -- before we trust the crosswalk's city.

Reads the flagged list from college_audit_flags.csv if present; otherwise uses a
built-in list of the genuinely ambiguous names.

Run:  python scripts/verify_players.py [N_per_school]
Output: printed table + college_verify_samples.csv
"""
import os
import sys
import pandas as pd

try:
    from college_crosswalk import normalize, lookup
except ImportError:
    print("ERROR: college_crosswalk.py must be importable.")
    sys.exit(1)

REG = 'data/tbc_draft_register.csv'
FLAGS = 'college_audit_flags.csv'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# the genuinely ambiguous ones worth manual lookup (not the false-alarm state flags)
PRIORITY = [
    'columbia', 'pacific', 'southern', 'concordia', 'concordia university',
    'saint thomas', 'st. thomas', 'trinity', 'trinity college', 'emory',
    'regis', 'dallas', 'union', 'union u', 'union commonwealth', 'lincoln',
    'miami', 'new york tech', 'mary',
]


def find(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def main():
    if not os.path.exists(REG):
        print(f"ERROR: {REG} not found.")
        sys.exit(1)
    reg = pd.read_csv(REG, low_memory=False)
    reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
    div = reg['schoolDivision'].astype(str)
    reg = reg[div.isin(['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA',
                        'NJCAA', 'CCCAA', 'NWAACC'])].copy()
    reg['k'] = reg['school'].apply(normalize)

    fn = find(reg, 'firstName')
    ln = find(reg, 'lastName')
    yr = find(reg, 'year')
    rnd = find(reg, 'draftRound')
    ov = find(reg, 'overall')
    place = find(reg, 'place')
    sdiv = find(reg, 'schoolDivision')
    signed = find(reg, 'signed')

    # which schools to sample
    schools = list(PRIORITY)
    if os.path.exists(FLAGS):
        fl = pd.read_csv(FLAGS)
        # add any flagged school not already in priority, biggest first
        for k in fl.sort_values('n_draftees', ascending=False)['key']:
            if k not in schools:
                schools.append(k)

    rows = []
    print("Look these players up (Baseball Reference / FanGraphs draft) to confirm")
    print("which school the register means, then tell me any that are wrong.\n")
    for sch in schools:
        sub = reg[reg['k'] == sch]
        if len(sub) == 0:
            continue
        loc = lookup(sch)
        loc_str = f"{loc[0]}, {loc[1]}" if loc else "(unresolved)"
        print("="*74)
        print(f"SCHOOL: '{sch}'  (n={len(sub)})  -> crosswalk says: {loc_str}")
        print("-"*74)
        # prefer signed players (easier to look up) and spread across years
        sub = sub.sort_values([signed, yr], ascending=[False, True]) if signed else sub
        for _, r in sub.head(N).iterrows():
            name = f"{r[fn]} {r[ln]}"
            print(f"  {name:<26} {int(r[yr])}  rd{int(r[rnd]) if pd.notna(r[rnd]) else 0:<3} "
                  f"ovr{int(r[ov]) if ov and pd.notna(r[ov]) else 0:<4} "
                  f"born: {str(r[place]) if place else '?':<20} {r[sdiv]}")
            rows.append({'school_key': sch, 'crosswalk_loc': loc_str,
                         'player': name, 'year': int(r[yr]),
                         'round': int(r[rnd]) if pd.notna(r[rnd]) else 0,
                         'overall': int(r[ov]) if ov and pd.notna(r[ov]) else 0,
                         'birth_place': str(r[place]) if place else '',
                         'division': r[sdiv]})
        print()

    pd.DataFrame(rows).to_csv('college_verify_samples.csv', index=False)
    print("Saved: college_verify_samples.csv")


if __name__ == '__main__':
    main()
