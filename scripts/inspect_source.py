"""
Inspect the RAW The Baseball Cube source register -- ground truth, not the build's
interpretation. Answers: what columns exist, what 'school'/'playerClass'/
'schoolDivision' actually hold, and what a COLLEGE draftee's row looks like vs
a HS draftee's -- specifically whether any HS location exists for college guys.

Run:  python scripts/inspect_source.py
"""
import pandas as pd

REG = 'data/tbc_draft_register.csv'

df = pd.read_csv(REG, low_memory=False)
print(f"File: {REG}")
print(f"Rows: {len(df):,}   Columns: {len(df.columns)}\n")

print("="*70)
print("ALL COLUMNS")
print("="*70)
for i, c in enumerate(df.columns):
    nn = df[c].notna().sum()
    print(f"  {i:>2} {c:<24} {nn/len(df)*100:>5.1f}% filled  "
          f"e.g. {repr(df[c].dropna().iloc[0]) if nn else '(empty)'}")

print("\n" + "="*70)
print("CLASS / DIVISION FIELDS -- how HS vs college is marked")
print("="*70)
for col in ['playerClass', 'schoolDivision', 'phase', 'draftType']:
    if col in df.columns:
        print(f"\n  {col} value counts:")
        print("   ", df[col].value_counts(dropna=False).head(12).to_dict())

print("\n" + "="*70)
print("THE 'school' FIELD -- HS draftee vs college draftee examples")
print("="*70)
# identify a class column
cls = next((c for c in ['playerClass', 'schoolDivision'] if c in df.columns), None)
schoolcol = next((c for c in ['school', 'School'] if c in df.columns), None)
if cls and schoolcol:
    hs_mask = df[cls].astype(str).str.upper().str.contains('HS|HIGH', na=False)
    print(f"\n  HS-class rows -- sample '{schoolcol}' values:")
    for v in df[hs_mask][schoolcol].dropna().head(6):
        print(f"    {repr(v)}")
    print(f"\n  NON-HS-class rows -- sample '{schoolcol}' values:")
    for v in df[~hs_mask][schoolcol].dropna().head(6):
        print(f"    {repr(v)}")

print("\n" + "="*70)
print("ANY HS-SPECIFIC COLUMN THAT SURVIVES FOR COLLEGE PLAYERS?")
print("="*70)
hs_cols = [c for c in df.columns if 'hs' in c.lower() or 'high' in c.lower()]
col_cols = [c for c in df.columns if 'college' in c.lower() or 'univ' in c.lower()]
print(f"  HS-ish columns: {hs_cols or 'NONE'}")
print(f"  college-ish columns: {col_cols or 'NONE'}")
print("\n  If there's no HS-specific column, a college row carries only the")
print("  college in 'school' -- confirming HS location is absent for college-only")
print("  draftees, and birth->college is the only supportable college-mover metric.")
