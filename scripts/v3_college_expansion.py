"""
Step 14 — College-draftee expansion.

The core analysis (steps 1-8) is HS draftees only, because the register's
`school` field stores the COLLEGE for college draftees, not their high school.
But `tbc_signing_bonus.csv` has `hsplace` (high-school location) for
ALL draftees, including college players. Merging it in lets us test the
mover effect on the full draft pool.

The key new question: does birth-to-HS mobility still predict reaching MLB
for players who went to college first (the "washout" test)?

What this script does:
  1. Load the register (birth city via `place`) + signing-bonus (`hsplace`).
  2. Merge on Baseball Cube player id; build a unified HS location for everyone.
  3. Classify HS vs college draftees.
  4. Reuse geocode_cache_v3.json (+ geocode any new HS towns inline) and compute
     birth-to-HS distance and the mover flag.
  5. Compare mover/stayer MLB reach rates for HS, college, and combined samples,
     and run a logistic model with a mover x college interaction.

Inputs:
  data/tbc_draft_register.csv
  data/tbc_signing_bonus.csv
  geocode_cache_v3.json        (reused; updated in place if new towns found)

Outputs:
  v3_analysis_expanded.csv             full HS+college sample with features
  v3_college_expansion_summary.csv     mover/stayer rates by group
  v3_college_interaction_logit.csv     regression incl. mover x college

Run:  python scripts/v3_college_expansion.py
"""

import os
import sys
import re
import json
import time
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from scipy import stats

REGISTER = 'data/tbc_draft_register.csv'
SIGNING  = 'data/tbc_signing_bonus.csv'
CACHE_FILE = 'geocode_cache_v3.json'

from config import START_YEAR, END_YEAR, COHORT_LABEL

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
    'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
    'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
    'VA','WA','WV','WI','WY','DC'
}
CA_PROVS = {'ON','QC','BC','AB','MB','SK','NS','NB','NL','PE','YT','NT','NU'}
COUNTRY_MAP = {
    'PR': 'Puerto Rico', 'DR': 'Dominican Republic', 'VE': 'Venezuela',
    'VZ': 'Venezuela', 'MX': 'Mexico', 'CU': 'Cuba', 'CB': 'Cuba',
    'JA': 'Japan', 'JP': 'Japan', 'KO': 'South Korea', 'AU': 'Australia',
    'AS': 'American Samoa', 'VI': 'US Virgin Islands', 'GU': 'Guam',
    'JM': 'Jamaica', 'PN': 'Panama', 'NI': 'Nicaragua', 'BZ': 'Belize',
}


# ── parsing helpers ───────────────────────────────────────────────────────────
def parse_place(p):
    """'San Diego,CA' -> ('San Diego', 'CA')"""
    if pd.isna(p):
        return None, None
    p = str(p).strip()
    if ',' in p:
        a, b = p.rsplit(',', 1)
        return a.strip(), b.strip()
    return p, None


def parse_school(s):
    """'Eastlake (Chula Vista,CA)' -> ('Eastlake', 'Chula Vista', 'CA')"""
    if pd.isna(s):
        return None, None, None
    s = str(s).strip()
    m = re.search(r'\(([^()]+)\)\s*$', s)
    if not m:
        return s, None, None
    inside = m.group(1).strip()
    name = s[:m.start()].strip()
    if ',' in inside:
        a, b = inside.rsplit(',', 1)
        return name, a.strip(), b.strip()
    return name, inside, None


def find_col(df, *candidates):
    """Return the first matching column name (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


# ── geocoding (reuse cache, fill gaps inline) ─────────────────────────────────
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def build_query(city, state):
    if not city or not state:
        return None
    city, state = str(city).strip(), str(state).strip()
    if state in US_STATES:
        return f"{city}, {state}, USA"
    if state in CA_PROVS:
        return f"{city}, {state}, Canada"
    if state in COUNTRY_MAP:
        return f"{city}, {COUNTRY_MAP[state]}"
    return f"{city}, {state}"


def geocode_one(query):
    try:
        import requests
        r = requests.get('https://nominatim.openstreetmap.org/search',
                         params={'q': query, 'format': 'json', 'limit': 1},
                         headers={'User-Agent': 'DraftMobilityResearch/1.0'},
                         timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        return {'lat': float(data[0]['lat']), 'lon': float(data[0]['lon']),
                'display_name': data[0]['display_name'], 'status': 'OK'}
    except Exception:
        return None


def coords_for(cache, city, state):
    if pd.isna(city) or pd.isna(state):
        return np.nan, np.nan
    key = f"{str(city).strip()}|{str(state).strip()}"
    v = cache.get(key)
    if v and v.get('lat') is not None:
        return v['lat'], v['lon']
    return np.nan, np.nan


def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return np.nan
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    d = sin((lat2-lat1)/2)**2 + cos(lat1)*cos(lat2)*sin((lon2-lon1)/2)**2
    return R * 2 * atan2(sqrt(d), sqrt(1-d))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(REGISTER):
        print(f"ERROR: {REGISTER} not found.")
        sys.exit(1)
    if not os.path.exists(SIGNING):
        print(f"ERROR: {SIGNING} not found (needed for college HS locations).")
        sys.exit(1)

    # 1. Register (base): birth city + classification
    reg = pd.read_csv(REGISTER, low_memory=False)
    reg = reg[(reg['year'] >= START_YEAR) & (reg['year'] <= END_YEAR)].copy()
    print(f"Register rows {START_YEAR}-{END_YEAR}: {len(reg):,}")

    reg[['birth_city', 'birth_state']] = reg['place'].apply(
        lambda p: pd.Series(parse_place(p)))
    reg[['reg_school_name', 'reg_school_city', 'reg_school_state']] = reg['school'].apply(
        lambda s: pd.Series(parse_school(s)))

    reg['reached_mlb'] = (reg['highLevel'] == 'MLB').astype(int)
    reg['signed_flag'] = (reg.get('signed', 'N').astype(str).str.strip() == 'Y').astype(int)
    reg['is_hs_draftee'] = ((reg.get('playerClass') == 'HS') |
                            (reg.get('schoolDivision') == 'HS')).astype(int)

    reg_pid = find_col(reg, 'PlayerID', 'playerid')
    reg_mlbid = find_col(reg, 'mlbid')
    print(f"Register key columns: id={reg_pid}, mlbid={reg_mlbid}")

    # 2. Signing-bonus: hsplace for everyone (incl. college)
    sb = pd.read_csv(SIGNING, low_memory=False)
    sb_pid = find_col(sb, 'playerid', 'PlayerID')
    sb_mlbid = find_col(sb, 'mlbid')
    sb_hsplace = find_col(sb, 'hsplace')
    sb_hsname = find_col(sb, 'hsName', 'hsname')
    if sb_hsplace is None:
        print("ERROR: no 'hsplace' column in signing-bonus file.")
        sys.exit(1)
    print(f"Signing-bonus rows: {len(sb):,}; key={sb_pid}, hsplace={sb_hsplace}")

    sb[['sb_hs_city', 'sb_hs_state']] = sb[sb_hsplace].apply(
        lambda p: pd.Series(parse_place(p)))

    # Build lookup maps keyed by player id and by mlbid
    hs_by_pid, hs_by_mlbid = {}, {}
    for _, r in sb.iterrows():
        city, state = r['sb_hs_city'], r['sb_hs_state']
        if pd.isna(city) or pd.isna(state):
            continue
        if sb_pid and pd.notna(r[sb_pid]):
            hs_by_pid[str(r[sb_pid]).strip()] = (city, state,
                                                 r[sb_hsname] if sb_hsname else None)
        if sb_mlbid and pd.notna(r[sb_mlbid]) and str(r[sb_mlbid]).strip() not in ('0', ''):
            hs_by_mlbid[str(r[sb_mlbid]).strip()] = (city, state,
                                                     r[sb_hsname] if sb_hsname else None)
    print(f"  hsplace available for {len(hs_by_pid):,} players (by id)")

    # 3. Unified HS location for every register row
    #    Priority: signing-bonus hsplace (works for college) -> register school city (HS only)
    hs_city, hs_state, hs_name, hs_src = [], [], [], []
    for _, r in reg.iterrows():
        c = s = nm = None
        src = 'none'
        # try signing-bonus by player id, then mlbid
        key_pid = str(r[reg_pid]).strip() if reg_pid and pd.notna(r[reg_pid]) else None
        key_mlb = str(r[reg_mlbid]).strip() if reg_mlbid and pd.notna(r[reg_mlbid]) else None
        if key_pid and key_pid in hs_by_pid:
            c, s, nm = hs_by_pid[key_pid]; src = 'signing_bonus'
        elif key_mlb and key_mlb in hs_by_mlbid:
            c, s, nm = hs_by_mlbid[key_mlb]; src = 'signing_bonus'
        elif r['is_hs_draftee'] == 1 and pd.notna(r['reg_school_city']):
            c, s, nm = r['reg_school_city'], r['reg_school_state'], r['reg_school_name']
            src = 'register_school'
        hs_city.append(c); hs_state.append(s); hs_name.append(nm); hs_src.append(src)

    reg['hs_city'] = hs_city
    reg['hs_state'] = hs_state
    reg['hs_name'] = hs_name
    reg['hs_source'] = hs_src

    reg['is_college_draftee'] = ((reg['is_hs_draftee'] == 0) &
                                 reg['hs_city'].notna()).astype(int)

    print(f"\nHS-location source breakdown:")
    print(reg['hs_source'].value_counts().to_string())

    # 4. Keep rows with both birth and HS location
    full = reg[reg['birth_city'].notna() & reg['birth_state'].notna() &
               reg['hs_city'].notna() & reg['hs_state'].notna()].copy()
    print(f"\nRows with BOTH birth & HS location: {len(full):,}")
    print(f"  HS draftees:      {int((full['is_hs_draftee']==1).sum()):,}")
    print(f"  College draftees: {int((full['is_college_draftee']==1).sum()):,}")

    # 5. Geocode (reuse cache; fill new towns inline)
    cache = load_cache()
    print(f"\nGeocode cache entries: {len(cache):,}")
    needed = set()
    for _, r in full.iterrows():
        needed.add((str(r['birth_city']).strip(), str(r['birth_state']).strip()))
        needed.add((str(r['hs_city']).strip(), str(r['hs_state']).strip()))
    missing = [(c, s) for (c, s) in needed if f"{c}|{s}" not in cache or
               cache.get(f"{c}|{s}", {}).get('lat') is None]
    print(f"Unique places needed: {len(needed):,}; not yet cached: {len(missing):,}")

    if missing:
        print(f"Geocoding {len(missing):,} new places (~{len(missing)/60:.1f} min)...")
        for i, (c, s) in enumerate(missing, 1):
            q = build_query(c, s)
            res = geocode_one(q) if q else None
            key = f"{c}|{s}"
            if res:
                cache[key] = res
            else:
                prior = int(cache.get(key, {}).get('attempts', 0))
                cache[key] = {'lat': None, 'lon': None, 'status': 'failed',
                              'query': q, 'attempts': prior + 1}
            if q:
                time.sleep(1.0)
            if i % 50 == 0:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache, f)
                print(f"  {i}/{len(missing)}")
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)

    # attach coords + distance
    bl = full.apply(lambda r: coords_for(cache, r['birth_city'], r['birth_state']),
                    axis=1, result_type='expand')
    full['birth_lat'], full['birth_lon'] = bl[0], bl[1]
    hl = full.apply(lambda r: coords_for(cache, r['hs_city'], r['hs_state']),
                    axis=1, result_type='expand')
    full['hs_lat'], full['hs_lon'] = hl[0], hl[1]

    full['distance_miles'] = [haversine(a, b, c, d) for a, b, c, d in
                              zip(full['birth_lat'], full['birth_lon'],
                                  full['hs_lat'], full['hs_lon'])]
    full['mover'] = np.where(full['distance_miles'].isna(), np.nan,
                             (full['distance_miles'] > 5).astype(float))

    have_dist = full['distance_miles'].notna()
    print(f"\nRows with distance computed: {have_dist.sum():,} "
          f"({have_dist.mean()*100:.1f}%)")

    full.to_csv('v3_analysis_expanded.csv', index=False, encoding='utf-8-sig')
    print("Saved: v3_analysis_expanded.csv")

    # 5b. Flattened ONE-ROW-PER-PERSON table (dedup multi-draft players).
    #     A player drafted out of HS who didn't sign, went to college, and was
    #     re-drafted appears as 2+ rows. Here we collapse to one record per
    #     person: reached_mlb = did this PERSON ever reach MLB (max across rows);
    #     represent him by his FIRST (HS) appearance when he has one, to preserve
    #     the original adolescent-era mobility signal. Flag players who appear as
    #     both HS and college draftees.
    pid_col = reg_pid  # Baseball Cube PlayerID, stable across a player's drafts
    flat = None
    if pid_col and pid_col in full.columns:
        f = full[full[pid_col].notna()].copy()
        f['_is_hs'] = f['is_hs_draftee'].astype(int)
        f['_is_col'] = f['is_college_draftee'].astype(int)
        # ever-reached MLB and both-path flag, per person
        ever = f.groupby(pid_col).agg(
            ever_mlb=('reached_mlb', 'max'),
            any_hs=('_is_hs', 'max'),
            any_college=('_is_col', 'max'),
            n_drafts=('reached_mlb', 'size')).reset_index()
        ever['appears_both'] = ((ever['any_hs'] == 1) & (ever['any_college'] == 1)).astype(int)

        # representative row: prefer the HS-draft appearance, else earliest year
        f = f.sort_values([pid_col, '_is_hs', 'year'], ascending=[True, False, True])
        rep = f.groupby(pid_col, as_index=False).first()

        flat = rep.merge(ever[[pid_col, 'ever_mlb', 'appears_both', 'n_drafts',
                               'any_hs', 'any_college']], on=pid_col, how='left')
        # person-level outcome overrides the per-row one
        flat['reached_mlb'] = flat['ever_mlb'].astype(int)
        # person-level path label: classify by representative (HS if ever HS)
        flat['person_path'] = np.where(flat['any_hs'] == 1, 'HS', 'College')

        flat.to_csv('v3_analysis_expanded_byperson.csv', index=False, encoding='utf-8-sig')
        n_both = int(flat['appears_both'].sum())
        print(f"\nFlattened to one row per person: {len(flat):,} players "
              f"(from {len(full):,} draft-event rows)")
        print(f"  Players appearing as BOTH HS and college draftees: {n_both:,}")
        print("Saved: v3_analysis_expanded_byperson.csv")
    else:
        print("\n(Could not build per-person table: no stable PlayerID column)")

    # 6. Mover/stayer comparison by group
    def rate_block(sub, label):
        sub = sub[sub['mover'].notna()]
        mv = sub[sub['mover'] == 1]
        st = sub[sub['mover'] == 0]
        if len(mv) == 0 or len(st) == 0:
            return None
        from scipy import stats
        ct = pd.crosstab(sub['mover'], sub['reached_mlb'])
        chi2 = p = np.nan
        if ct.shape == (2, 2):
            chi2, p, _, _ = stats.chi2_contingency(ct)
        return {
            'group': label, 'n': len(sub),
            'n_movers': len(mv), 'n_stayers': len(st),
            'mover_pct': round(mv['reached_mlb'].mean()*100, 2),
            'stayer_pct': round(st['reached_mlb'].mean()*100, 2),
            'diff_pp': round((mv['reached_mlb'].mean()-st['reached_mlb'].mean())*100, 2),
            'chi2': round(chi2, 2) if not np.isnan(chi2) else None,
            'p_value': p,
        }

    print(f"\n{'='*78}")
    print("MOVER vs STAYER MLB REACH RATE, BY GROUP")
    print(f"{'='*78}")
    rows = []
    for sub, lab in [(full[full['is_hs_draftee'] == 1], 'HS draftees'),
                     (full[full['is_college_draftee'] == 1], 'College draftees'),
                     (full, 'Combined (HS + college)')]:
        rb = rate_block(sub, lab)
        if rb:
            rows.append(rb)
            print(f"\n{lab}  (n={rb['n']:,})")
            print(f"  Movers:  {rb['mover_pct']:.1f}%  (n={rb['n_movers']:,})")
            print(f"  Stayers: {rb['stayer_pct']:.1f}%  (n={rb['n_stayers']:,})")
            print(f"  Diff:    {rb['diff_pp']:+.1f} pp   chi2={rb['chi2']}  p={rb['p_value']:.2e}")
    pd.DataFrame(rows).to_csv('v3_college_expansion_summary.csv', index=False)

    # 6b. Same comparison on the DEDUPED per-person table
    if flat is not None:
        print(f"\n{'='*78}")
        print("MOVER vs STAYER — DEDUPED (one row per person; ever reached MLB)")
        print(f"{'='*78}")
        prows = []
        for sub, lab in [(flat[flat['person_path'] == 'HS'], 'HS path'),
                         (flat[flat['person_path'] == 'College'], 'College-only path'),
                         (flat, 'All players')]:
            rb = rate_block(sub, lab)
            if rb:
                prows.append(rb)
                print(f"\n{lab}  (n={rb['n']:,})")
                print(f"  Movers:  {rb['mover_pct']:.1f}%  (n={rb['n_movers']:,})")
                print(f"  Stayers: {rb['stayer_pct']:.1f}%  (n={rb['n_stayers']:,})")
                print(f"  Diff:    {rb['diff_pp']:+.1f} pp   chi2={rb['chi2']}  p={rb['p_value']:.2e}")
        pd.DataFrame(prows).to_csv('v3_college_expansion_summary_byperson.csv', index=False)
        print("\nSaved: v3_college_expansion_summary_byperson.csv")

    # 7. Logistic model with mover x college interaction
    try:
        import statsmodels.api as sm
        m = full[full['mover'].notna() & full['distance_miles'].notna()].copy()
        m['year_c'] = m['year'] - 2010
        m['rounds_1_5'] = (m['draftRound'] <= 5).astype(int)
        m['rounds_6_10'] = ((m['draftRound'] >= 6) & (m['draftRound'] <= 10)).astype(int)
        m['rounds_11_20'] = ((m['draftRound'] >= 11) & (m['draftRound'] <= 20)).astype(int)
        m['college'] = m['is_college_draftee'].astype(int)
        m['mover_x_college'] = m['mover'] * m['college']

        feats = ['mover', 'college', 'mover_x_college', 'year_c',
                 'rounds_1_5', 'rounds_6_10', 'rounds_11_20']
        X = sm.add_constant(m[feats])
        y = m['reached_mlb'].astype(int)
        res = sm.Logit(y, X).fit(disp=False, maxiter=200)

        print(f"\n{'='*78}")
        print("LOGISTIC REGRESSION — mover effect with college interaction")
        print(f"{'='*78}")
        print(f"N={int(res.nobs):,}  Pseudo R2={res.prsquared:.4f}")
        out = []
        for v in X.columns:
            if v == 'const':
                continue
            coef, p = res.params[v], res.pvalues[v]
            orr = np.exp(coef)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            tag = '  <<<' if v in ('mover', 'mover_x_college') else ''
            print(f"  {v:<18} coef={coef:+.4f}  OR={orr:6.3f}  p={p:.4f} {sig}{tag}")
            out.append({'variable': v, 'coef': round(coef, 5),
                        'odds_ratio': round(orr, 4), 'p_value': round(p, 6)})
        pd.DataFrame(out).to_csv('v3_college_interaction_logit.csv', index=False)

        # interpret the interaction
        mover_or_hs = np.exp(res.params['mover'])
        mover_or_col = np.exp(res.params['mover'] + res.params['mover_x_college'])
        print(f"\nMover odds ratio for HS draftees:      {mover_or_hs:.3f}")
        print(f"Mover odds ratio for college draftees: {mover_or_col:.3f}")
        ip = res.pvalues['mover_x_college']
        if ip < 0.05:
            print(f"Interaction significant (p={ip:.3f}): the mover effect DIFFERS by path.")
        else:
            print(f"Interaction NOT significant (p={ip:.3f}): the mover effect is similar "
                  f"for HS and college draftees (it survives the college washout).")

        # --- 7b. SIGNED-ONLY robustness -------------------------------------
        # High-school locations for COLLEGE draftees come from the signing-bonus
        # file, which only covers players who signed. The HS subsample also
        # includes unsigned players via the register's school field. That makes
        # the two groups differently selected. Re-running the interaction on
        # signed players only removes the asymmetry.
        if 'signed_flag' in m.columns:
            ms = m[m['signed_flag'] == 1].copy()
            n_hs = int((ms['college'] == 0).sum())
            n_col = int((ms['college'] == 1).sum())
            if n_hs > 100 and n_col > 100 and ms['reached_mlb'].nunique() > 1:
                Xs = sm.add_constant(ms[feats])
                ys = ms['reached_mlb'].astype(int)
                rs = sm.Logit(ys, Xs).fit(disp=False, maxiter=200)
                or_hs_s = np.exp(rs.params['mover'])
                or_col_s = np.exp(rs.params['mover'] + rs.params['mover_x_college'])
                ip_s = rs.pvalues['mover_x_college']
                print(f"\n{'-'*78}")
                print("SIGNED-ONLY ROBUSTNESS (both groups conditioned on signing)")
                print(f"{'-'*78}")
                print(f"N={int(rs.nobs):,}  (HS {n_hs:,} / college {n_col:,})")
                print(f"  Mover OR, HS draftees:      {or_hs_s:.3f}")
                print(f"  Mover OR, college draftees: {or_col_s:.3f}")
                print(f"  mover x college p = {ip_s:.4f}")
                if ip_s < 0.05:
                    print("  -> Washout HOLDS with selection equalized.")
                else:
                    print("  -> Interaction not significant once both groups are "
                          "signed-only; the washout may be a selection artifact.")
                pd.DataFrame([{
                    'sample': 'signed only', 'n': int(rs.nobs),
                    'n_hs': n_hs, 'n_college': n_col,
                    'mover_or_hs': round(or_hs_s, 4),
                    'mover_or_college': round(or_col_s, 4),
                    'interaction_p': ip_s,
                }]).to_csv('v3_college_signed_only.csv', index=False)
                print("  saved v3_college_signed_only.csv")
            else:
                print("\n(signed-only check skipped: subgroup too small)")
        else:
            print("\n(signed-only check skipped: no signed_flag column)")

    except Exception as e:
        print(f"\n(logit skipped: {e})")

    # 8. WAR-quality washout: does career VALUE (not just reach rate) wash out
    #    for college players? Join career WAR from the FanGraphs-joined file.
    war_src = 'v3_analysis_with_war.csv'
    if os.path.exists(war_src) and reg_mlbid:
        try:
            wdf = pd.read_csv(war_src, low_memory=False, usecols=lambda c: c in
                              ('mlbid', 'career_war', 'fg_any_match'))
            wdf = wdf[wdf['mlbid'].notna() & (wdf['mlbid'].astype(str) != '0')]
            wdf['mlbid_key'] = wdf['mlbid'].astype(str).str.strip()
            war_map = dict(zip(wdf['mlbid_key'], wdf['career_war']))

            f2 = full[full['mover'].notna()].copy()
            f2['mlbid_key'] = f2[reg_mlbid].astype(str).str.strip()
            f2['career_war'] = f2['mlbid_key'].map(war_map)
            # WAR per draftee: non-MLB / unmatched contribute 0 value
            f2['war_contrib'] = f2['career_war'].fillna(0.0)

            print(f"\n{'='*78}")
            print("WAR-QUALITY WASHOUT — mean career WAR per draftee, mover vs stayer")
            print(f"{'='*78}")
            wrows = []
            for sub, lab in [(f2[f2['is_hs_draftee'] == 1], 'HS draftees'),
                             (f2[f2['is_college_draftee'] == 1], 'College draftees'),
                             (f2, 'Combined')]:
                mv = sub[sub['mover'] == 1]['war_contrib']
                st = sub[sub['mover'] == 0]['war_contrib']
                if len(mv) == 0 or len(st) == 0:
                    continue
                tt, tp = stats.ttest_ind(mv, st, equal_var=False)
                print(f"\n{lab}  (n={len(sub):,})")
                print(f"  Movers:  {mv.mean():.3f} WAR/draftee  (n={len(mv):,})")
                print(f"  Stayers: {st.mean():.3f} WAR/draftee  (n={len(st):,})")
                print(f"  Diff:    {mv.mean()-st.mean():+.3f}   t={tt:.2f}  p={tp:.2e}")
                wrows.append({'group': lab, 'n': len(sub),
                              'mover_war_per_draftee': round(mv.mean(), 4),
                              'stayer_war_per_draftee': round(st.mean(), 4),
                              'diff': round(mv.mean()-st.mean(), 4),
                              't_stat': round(tt, 3), 'p_value': tp})
            pd.DataFrame(wrows).to_csv('v3_college_war_washout.csv', index=False)
            print("\nSaved: v3_college_war_washout.csv")
            print("Interpretation: if the mover WAR advantage shrinks from HS to college "
                  "(like reach rate did), the career-value signal washes out too.")
        except Exception as e:
            print(f"\n(WAR washout skipped: {e})")
    else:
        print(f"\n(WAR washout skipped: {war_src} not found — run step 6 first)")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
