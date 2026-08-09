"""
run_pipeline.py — single entry point for the Born to Move analysis pipeline.

The pipeline runs in stages. Each stage depends on the previous one's output, so
run them in order (or use --step all). You must supply the source data yourself
under data/ (see README) before running the build stage.

Usage:
    python run_pipeline.py --step STAGE

Stages:
    build     Build the base mover/stayer dataset from the raw draft records
              (data/tbc_draft_register.csv) -> v3_analysis.csv
    geocode   Geocode birth and high-school cities via Nominatim (cached; obeys
              the 1 request/second usage policy)
    distance  Compute great-circle birth->HS distances and the mover flag,
              writing back to v3_analysis.csv
    join      Join FanGraphs career stats (WAR, PA, IP) onto the dataset
              -> v3_analysis_with_war.csv
    dedup     Deduplicate players with multiple HS-draft rows (keep signed row,
              else latest) -> promotes the clean file as the analysis input
    analyze   Run all statistical analyses and regenerate every figure
              (this is the run_all_hs.py stage)
    all       Run every stage above, in order

Examples:
    python run_pipeline.py --step all
    python run_pipeline.py --step analyze     # if the dataset is already built

Notes:
    - Run from the repository root: python run_pipeline.py --step ...
    - Source data is NOT included in this repo (licensing). See README.
    - Geocoding is cached; the first full run is slow (one request/second).
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, "scripts")

# stage -> ordered list of scripts to run for that stage
STAGES = {
    "build":    ["build_v3_analysis.py"],
    "geocode":  ["v3_geocode.py"],
    "distance": ["v3_compute_distances.py"],
    "join":     ["v3_join_fangraphs.py"],
    "dedup":    ["dedup_hs_draftees.py", "promote_deduped.py"],
    "analyze":  ["run_all_hs.py"],
}
ORDER = ["build", "geocode", "distance", "join", "dedup", "analyze"]


def run_script(name):
    path = os.path.join(SCRIPTS, name)
    if not os.path.exists(path):
        print(f"  [skip] {name} not found")
        return True
    print(f"\n=== running {name} ===")
    # run from repo root so scripts find data/ and write outputs consistently
    r = subprocess.run([sys.executable, path], cwd=ROOT)
    if r.returncode != 0:
        print(f"  [error] {name} exited {r.returncode}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Born to Move analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages (in order): " + " -> ".join(ORDER))
    ap.add_argument("--step", required=True,
                    choices=ORDER + ["all"],
                    help="which stage to run (or 'all')")
    args = ap.parse_args()

    stages = ORDER if args.step == "all" else [args.step]
    for stage in stages:
        print(f"\n########## STAGE: {stage} ##########")
        for script in STAGES[stage]:
            if not run_script(script):
                print(f"\nStopped: stage '{stage}' failed at {script}.")
                sys.exit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
