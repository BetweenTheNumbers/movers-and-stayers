"""List every Concordia-variant player so they can be looked up manually to
determine WHICH Concordia (Irvine, Chicago, Portland, St. Paul, Texas, etc.).

Run:  python scripts/list_concordia.py
"""
import pandas as pd
from college_crosswalk import normalize

reg = pd.read_csv('data/tbc_draft_register.csv',
                  low_memory=False)
reg = reg[(reg['year'] >= 1996) & (reg['year'] <= 2019)].copy()
reg['k'] = reg['school'].apply(normalize)

# every distinct register spelling containing 'concordia'
concordia_keys = sorted(k for k in reg['k'].unique() if 'concordia' in str(k))
print("Distinct register spellings containing 'concordia':")
print("  " + ", ".join(concordia_keys) + "\n")

for key in concordia_keys:
    sub = reg[reg['k'] == key]
    print(f"=== '{key}'  (n={len(sub)}) ===")
    for _, r in sub.sort_values('year').iterrows():
        rnd = int(r['draftRound']) if pd.notna(r['draftRound']) else 0
        ov = int(r['overall']) if pd.notna(r['overall']) else 0
        print(f"  {r['firstName']} {r['lastName']:<20} {int(r['year'])}  "
              f"rd{rnd:<3} ovr{ov:<5} born: {str(r['place']):<22} "
              f"{r['schoolDivision']}  signed={r['signed']}")
    print()
