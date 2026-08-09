"""
Run the full HS analysis + figures against the CURRENT analysis input
(v3_analysis_with_war.csv) -- use this AFTER promote_deduped.py has made the
deduped data permanent, so no backup/swap is needed. Just runs everything in
order and reports pass/fail.

Run:  python scripts/run_all_hs.py
"""
import os
import sys
import subprocess

PY = sys.executable
SCRIPTDIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    # analysis (compute) -- order matters where one writes a CSV another reads
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
    'v3_player_records.py',
    # figures
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
        return 'skip'
    print(f"\n{'='*60}\nRUN: {script}\n{'='*60}")
    r = subprocess.run([PY, path], capture_output=True, text=True,
                       cwd=os.getcwd())
    if r.stdout:
        print(r.stdout.strip()[-2000:])
    if r.returncode != 0:
        print(f"  !! exited {r.returncode}")
        if r.stderr:
            print("  stderr tail:\n  " + '\n  '.join(r.stderr.splitlines()[-8:]))
        return 'fail'
    return 'ok'


def main():
    if not os.path.exists('v3_analysis_with_war.csv'):
        print("ERROR: v3_analysis_with_war.csv not found in current dir.")
        print("Run this from the project root (python scripts\\run_all_hs.py).")
        sys.exit(1)
    n = sum(1 for _ in open('v3_analysis_with_war.csv')) - 1
    print(f"Input: v3_analysis_with_war.csv ({n:,} rows)\n")

    results = {}
    for s in SCRIPTS:
        results[s] = run(s)

    ok = [s for s, v in results.items() if v == 'ok']
    fail = [s for s, v in results.items() if v == 'fail']
    skip = [s for s, v in results.items() if v == 'skip']
    print("\n" + "="*60)
    print(f"DONE: {len(ok)} ok, {len(fail)} failed, {len(skip)} skipped")
    print("="*60)
    if fail:
        print(f"  FAILED: {fail}")
    if skip:
        print(f"  skipped (not found): {skip}")
    if not fail:
        print("  Clean run. Fresh numbers + 32 figures in figures/.")


if __name__ == '__main__':
    main()
