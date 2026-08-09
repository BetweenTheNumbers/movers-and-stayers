"""
Audit college_locations_resolved.csv for LIKELY wrong matches, so the review is
~30 flagged rows instead of 1,280. Flags:

  1. Ambiguous short/common names (dallas, chicago, regis, columbia, emory,
     pacific, southern, trinity, concordia, st. *, etc.) -- these have multiple
     real schools and are the top wrong-match risk.
  2. Entries resolved only via the ALIAS fallback (not an exact key) -- the
     looser match is where over-collapsing can happen.
  3. State mismatches: resolved state != the state hinted in the school name
     (e.g. a name containing 'ohio' resolving to a non-OH city).
  4. Draftee-count weighted: shows n_draftees so you review high-volume first.

Read-only. Prints a review list; writes college_audit_flags.csv.

Run:  python scripts/audit_crosswalk.py
"""
import os
import sys
import re as _re
import pandas as pd

try:
    from college_crosswalk import CROSSWALK, normalize, alias_key
except ImportError:
    print("ERROR: college_crosswalk.py must be importable from here.")
    sys.exit(1)

RESOLVED = 'college_locations_resolved.csv'
UNMATCHED = 'college_unmatched.csv'
REG = 'data/tbc_draft_register.csv'

# names that are inherently ambiguous -> always worth an eyeball
AMBIGUOUS = {
    'dallas', 'chicago', 'regis', 'columbia', 'emory', 'pacific', 'southern',
    'trinity', 'concordia', 'miami', 'newman', 'friends', 'barton', 'mary',
    'rust', 'westmar', 'montmorency', 'patten college', 'st. katherine',
    'wesleyan university', 'williams', 'brandeis', 'new york', 'chesapeake',
    'central baptist', 'lincoln', 'union', 'wooster', 'oxford',
}

# state-name tokens that should appear in the resolved state if in the name
STATE_TOKENS = {
    'ohio': 'OH', 'texas': 'TX', 'florida': 'FL', 'california': 'CA',
    'georgia': 'GA', 'alabama': 'AL', 'tennessee': 'TN', 'kentucky': 'KY',
    'indiana': 'IN', 'illinois': 'IL', 'michigan': 'MI', 'iowa': 'IA',
    'kansas': 'KS', 'missouri': 'MO', 'nebraska': 'NE', 'arizona': 'AZ',
    'oregon': 'OR', 'washington': 'WA', 'colorado': 'CO', 'utah': 'UT',
    'nevada': 'NV', 'virginia': 'VA', 'carolina': None, 'dakota': None,
    'mississippi': 'MS', 'louisiana': 'LA', 'arkansas': 'AR', 'oklahoma': 'OK',
    'maryland': 'MD', 'pennsylvania': 'PA', 'wisconsin': 'WI', 'minnesota': 'MN',
}


def main():
    if not os.path.exists(RESOLVED):
        print(f"ERROR: {RESOLVED} not found. Run resolve_colleges.py first.")
        sys.exit(1)
    res = pd.read_csv(RESOLVED)
    # draftee counts (from unmatched-style groupby on the register)
    counts = {}
    if os.path.exists(REG):
        reg = pd.read_csv(REG, low_memory=False)
        reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)]
        div = reg['schoolDivision'].astype(str)
        college = reg[div.isin(['NCAA 1', 'NCAA 2', 'NCAA 3', 'NAIA',
                                'NJCAA', 'CCCAA', 'NWAACC'])].copy()
        college['k'] = college['school'].apply(normalize)
        counts = college['k'].value_counts().to_dict()

    res['n'] = res['key'].map(counts).fillna(0).astype(int)
    res['exact'] = res['key'].apply(lambda k: normalize(k) in CROSSWALK)

    flags = []
    for _, r in res.iterrows():
        key = str(r['key'])
        city = str(r.get('city', ''))
        state = str(r.get('state', ''))
        reasons = []
        # 1. ambiguous name
        if key in AMBIGUOUS or any(key.startswith(a + ' ') or key == a
                                   for a in AMBIGUOUS):
            reasons.append('ambiguous-name')
        # 2. alias matches are NOT flagged alone -- too noisy, and by-design.
        # 3. state token mismatch (word-boundary, not substring, to avoid
        #    'kansas' matching inside 'arkansas' etc.)
        for tok, st in STATE_TOKENS.items():
            if st and state and state != st and \
               _re.search(r'\b' + tok + r'\b', key):
                reasons.append(f'state? name~{tok} but resolved {state}')
                break
        if reasons:
            flags.append({'n_draftees': r['n'], 'key': key,
                          'city': city, 'state': state,
                          'flags': '; '.join(reasons)})

    fdf = pd.DataFrame(flags).sort_values('n_draftees', ascending=False)
    fdf.to_csv('college_audit_flags.csv', index=False)

    print("="*70)
    print(f"AUDIT: {len(res):,} resolved schools, {len(fdf):,} flagged for review")
    print("="*70)
    print(f"{'n':>4}  {'school':<32}{'-> city, ST':<26} why")
    for _, r in fdf.head(40).iterrows():
        loc = f"{r['city']}, {r['state']}"
        print(f"{r['n_draftees']:>4}  {r['key'][:32]:<32}{loc[:26]:<26} {r['flags']}")
    print(f"\nFull list: college_audit_flags.csv ({len(fdf)} rows)")
    print("Review these; the other", len(res)-len(fdf), "are exact unambiguous matches.")


if __name__ == '__main__':
    main()
