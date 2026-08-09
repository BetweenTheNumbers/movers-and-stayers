"""
Re-run the FULL HS mover analysis + all figures on the DEDUPED data, using the
cached geocodes. Does NOT re-run upstream builders (build_v3_analysis /
v3_compute_distances / v3_join_fangraphs) -- those would rebuild from the raw
register and re-introduce the 58 duplicates. Instead we swap the deduped file in
as the analysis input, run every downstream analysis + figure script, then
restore the original.

Safe: backs up v3_analysis_with_war.csv -> v3_analysis_with_war.PREDEDUP.csv and
restores it at the end (even on error).

Run:  python scripts/rerun_hs_fresh.py
"""
import os
import sys
import shutil
import subprocess

ANALYSIS = 'v3_analysis_with_war.csv'
DEDUPED = 'v3_analysis_hs_deduped.csv'
BACKUP = 'v3_analysis_with_war.PREDEDUP.csv'
PY = sys.executable

# scripts live in scripts/ ; this runner is invoked from the project root
# (python scripts\rerun_hs_fresh.py), so prefix the folder when calling them.
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))

# downstream scripts to run, in dependency order. Analysis (compute) first so any
# intermediate CSVs exist before the figure scripts that plot them.
ANALYSIS_SCRIPTS = [
    'v3_mover_tiers.py',
    'v3_mover_tiers_bytype.py',
    'v3_war_analysis.py',
    'v3_season_rates.py',
    'v3_window_rates.py',
    'v3_runway_bias.py',
    'v3_temporal_trend.py',
    'v3_mover_share_trend.py',
    'v3_threshold_sensitivity.py',
    'v3_year_window_test.py',
    'v3_distance_buckets.py',
    'v3_tier_full_value.py',
    'v3_tier_windows.py',
    'v3_tier_matched_thresh.py',
    'v3_batarm_trend.py',
    'v3_market_efficiency.py',
    'v3_coverage_gap.py',
    'v3_war_analysis.py',
    'v3_player_records.py',
]
FIGURE_SCRIPTS = [
    'v3_make_figures.py',
    'v3_make_distance_figures.py',
    'v3_make_distance_permile.py',
    'v3_make_geo_figures.py',
    'v3_make_metro_maps.py',
    'v3_make_quality_heatmaps.py',
    'v3_make_figure_index.py',
]


def run(script):
    path = os.path.join(SCRIPTDIR, script)
    if not os.path.exists(path):
        print(f"  SKIP (missing): {script}")
        return None
    print(f"\n{'='*60}\nRUN: {script}\n{'='*60}")
    # scripts read data files (v3_analysis_with_war.csv, geocode cache, etc.) by
    # RELATIVE path, so run them with the project root as cwd -- that's the dir
    # containing the analysis file, i.e. where THIS runner was invoked from.
    r = subprocess.run([PY, path], capture_output=True, text=True, cwd=os.getcwd())
    out = (r.stdout or '').strip()
    err = (r.stderr or '').strip()
    if out:
        print(out[-2000:])          # tail of stdout
    if r.returncode != 0:
        print(f"  !! {script} exited {r.returncode}")
        if err:
            print("  stderr tail:\n  " + '\n  '.join(err.splitlines()[-8:]))
    return r.returncode


def main():
    if not os.path.exists(DEDUPED):
        print(f"ERROR: {DEDUPED} not found. Run dedup_hs_draftees.py first.")
        sys.exit(1)

    # backup original, swap in deduped
    if os.path.exists(ANALYSIS):
        shutil.copy2(ANALYSIS, BACKUP)
        print(f"Backed up {ANALYSIS} -> {BACKUP}")
    shutil.copy2(DEDUPED, ANALYSIS)
    print(f"Swapped deduped data in as {ANALYSIS} "
          f"({sum(1 for _ in open(DEDUPED))-1:,} rows)\n")

    failed = []
    try:
        print("\n########## ANALYSIS SCRIPTS ##########")
        for s in ANALYSIS_SCRIPTS:
            rc = run(s)
            if rc not in (0, None):
                failed.append(s)
        print("\n########## FIGURE SCRIPTS ##########")
        for s in FIGURE_SCRIPTS:
            rc = run(s)
            if rc not in (0, None):
                failed.append(s)
    finally:
        # restore original no matter what
        if os.path.exists(BACKUP):
            shutil.copy2(BACKUP, ANALYSIS)
            print(f"\nRestored original {ANALYSIS} from {BACKUP}")

    print("\n" + "="*60)
    print("RERUN COMPLETE")
    print("="*60)
    if failed:
        print(f"  {len(failed)} script(s) had errors: {failed}")
        print("  (others completed; check their output above)")
    else:
        print("  all scripts completed with no errors")
    print("  Figures refreshed in figures/  |  original analysis file restored")
    print("\n  NOTE: the deduped run overwrote result CSVs + figures with the")
    print("  cleaned versions. The analysis INPUT file was restored to original,")
    print("  but the OUTPUT csvs/pngs now reflect the deduped data (that's what")
    print("  you wanted -- fresh numbers). To make deduped the permanent input,")
    print("  copy v3_analysis_hs_deduped.csv over v3_analysis_with_war.csv.")


if __name__ == '__main__':
    main()
