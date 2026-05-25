"""
Run all experiments sequentially and produce a summary.

Usage:
    python run_all.py [--synthetic-only] [--fred-only]
"""
import argparse
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--synthetic-only", action="store_true")
parser.add_argument("--fred-only", action="store_true")
args = parser.parse_args()

def run(module: str):
    print(f"\n{'#'*70}")
    print(f"# Running {module}")
    print(f"{'#'*70}\n")
    result = subprocess.run([sys.executable, "-m", module], check=True)
    return result

if not args.fred_only:
    run("experiments.exp_synthetic")

if not args.synthetic_only:
    run("experiments.exp_fred")

print("\nAll experiments complete. Results are in ./results/ and figures in ./figures/")
