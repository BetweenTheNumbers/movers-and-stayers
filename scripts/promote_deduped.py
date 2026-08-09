"""
Make the deduped HS data the PERMANENT analysis input.
  - backs up the current v3_analysis_with_war.csv -> v3_analysis_with_war.ORIG.csv
    (only if that backup doesn't already exist, so we never clobber the true
    original on a second run)
  - copies v3_analysis_hs_deduped.csv over v3_analysis_with_war.csv

After this, every analysis/figure script reads the clean deduped data by default.

Run:  python scripts/promote_deduped.py
"""
import os
import shutil
import sys

ANALYSIS = 'v3_analysis_with_war.csv'
DEDUPED = 'v3_analysis_hs_deduped.csv'
ORIG = 'v3_analysis_with_war.ORIG.csv'


def main():
    if not os.path.exists(DEDUPED):
        print(f"ERROR: {DEDUPED} not found. Run dedup_hs_draftees.py first.")
        sys.exit(1)

    # preserve the true original once
    if os.path.exists(ANALYSIS) and not os.path.exists(ORIG):
        shutil.copy2(ANALYSIS, ORIG)
        print(f"Preserved original -> {ORIG}")
    elif os.path.exists(ORIG):
        print(f"Original already preserved at {ORIG} (leaving it untouched)")

    shutil.copy2(DEDUPED, ANALYSIS)
    n = sum(1 for _ in open(ANALYSIS)) - 1
    print(f"Promoted: {DEDUPED} is now {ANALYSIS} ({n:,} rows).")
    print("All analysis/figure scripts will now read the clean deduped data.")
    print(f"\nTo revert if ever needed: copy {ORIG} back over {ANALYSIS}.")


if __name__ == '__main__':
    main()
