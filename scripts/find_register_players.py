"""
Identify the players behind a list of unresolved (city, state) pairs, scanning
the FULL draft register (1965-2025), not just the cohort analysis file.

Use this for the historical geocode failures that geocode_all_years.py logs --
unresolved_players.py only checks the cohort cache, so it misses pre-1996 towns.

Usage:
    python scripts/find_register_players.py geocode_failures.csv

Reads the failure CSV (needs 'city','state' columns), then prints every
register row whose birth place OR high-school city matches an unresolved pair,
with name, year, team, round. Writes register_players_for_failures.csv.
"""

import re
import sys
import pandas as pd

REGISTER = 'data/tbc_draft_register.csv'


def parse_place(p):
    if pd.isna(p) or ',' not in str(p):
        return None, None
    a, b = str(p).rsplit(',', 1)
    return a.strip(), b.strip()


def parse_school_city(s):
    if pd.isna(s):
        return None, None
    m = re.search(r'\(([^()]+)\)\s*$', str(s))
    if not m or ',' not in m.group(1):
        return None, None
    a, b = m.group(1).rsplit(',', 1)
    return a.strip(), b.strip()


def main():
    if len(sys.argv) < 2:
        print("usage: python scripts/find_register_players.py geocode_failures.csv")
        sys.exit(1)
    fails = pd.read_csv(sys.argv[1])
    fails.columns = [c.strip().lower() for c in fails.columns]
    want = {(str(r['city']).strip(), str(r['state']).strip())
            for _, r in fails.iterrows()}
    print(f"Looking for {len(want)} unresolved (city,state) pairs\n")

    reg = pd.read_csv(REGISTER, low_memory=False)
    bp = reg['place'].apply(lambda p: pd.Series(parse_place(p)))
    reg['bc'], reg['bs'] = bp[0], bp[1]
    sp = reg['school'].apply(lambda s: pd.Series(parse_school_city(s)))
    reg['hc'], reg['hs'] = sp[0], sp[1]

    fn = 'firstName' if 'firstName' in reg.columns else None
    ln = 'lastName' if 'lastName' in reg.columns else None
    team = next((c for c in ['Teamname', 'team', 'Team'] if c in reg.columns), None)
    rnd = 'draftRound' if 'draftRound' in reg.columns else None

    rows = []
    for _, r in reg.iterrows():
        hits = []
        if (str(r['bc']).strip(), str(r['bs']).strip()) in want:
            hits.append(f"birth={r['bc']}, {r['bs']}")
        if (str(r['hc']).strip(), str(r['hs']).strip()) in want:
            hits.append(f"school={r['hc']}, {r['hs']}")
        if not hits:
            continue
        rows.append({
            'player': f"{r.get(fn,'')} {r.get(ln,'')}".strip(),
            'year': r.get('year', ''), 'team': r.get(team, ''),
            'round': r.get(rnd, ''),
            'reached': r.get('highLevel', ''),
            'unresolved': '; '.join(hits),
        })

    if not rows:
        print("No register rows matched these pairs.")
        return
    out = pd.DataFrame(rows).sort_values('unresolved')
    out.to_csv('register_players_for_failures.csv', index=False,
               encoding='utf-8-sig')
    print(f"Matched {len(out)} register rows across "
          f"{out['unresolved'].nunique()} unresolved locations.")
    print("Saved register_players_for_failures.csv")


if __name__ == '__main__':
    main()
