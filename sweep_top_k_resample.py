"""
Sweeps globals.top_k_resample across a few values and computes core metrics
(nDCG, behavioral ILD, geographic ILD) for baseline/platform/provider/civic at
each one, so you can see where diversity gains plateau relative to relevance
loss and pick a candidate-pool size on evidence rather than a guess.

Scope, deliberately kept small for a first pass:
  - Only reruns the stages that actually read top_k_resample: platform_reranker.py,
    provider_reranker.py, civic_reranker.py (postprocess_baseline_top_k.py and
    compute_user_compatibility.py don't use it -- run those once beforehand via
    run_rerank_and_aggregate.sh if you haven't already).
  - Skips social_choice_aggregation.py / kemeny_young for now -- add it back
    into STAGES below once this smaller sweep is validated.
  - Values must stay <=150: recbole_full_casestudy.py hardcodes k=150 for the
    raw RecBole candidate pool, so anything larger needs that bumped and the
    model rerun first.

Each value's output directory is snapshotted (copied) to a `__topk<N>`
suffixed sibling before the next value overwrites it in place, since the
reranker scripts always write to the same fixed-name method subdirectories.

Usage:
    python3 sweep_top_k_resample.py --values 30 50 100 150
"""

import argparse
import os
import shutil
import subprocess
import sys

import pandas as pd

from compute_sweep_metrics import compute_metrics_for_run
from globals import BASE_DIR, PROJECT_BASE, available_datasets, recommendation_dirpart
from platform_reranker import dataset_metadata

STAGES = ["platform_reranker.py", "provider_reranker.py", "civic_reranker.py"]
RESULTS_DIR = os.path.join(PROJECT_BASE, "sweeps")


def run_stage(script, value):
    env = os.environ.copy()
    env["TOP_K_RESAMPLE"] = str(value)
    print(f"$ TOP_K_RESAMPLE={value} python3 {script}")
    subprocess.run([sys.executable, script], check=True, env=env, cwd=PROJECT_BASE)


def snapshot_and_measure(value):
    all_rows = []
    for dataset in available_datasets:
        for result in dataset_metadata(dataset):
            model_dir = os.path.join(
                BASE_DIR, f"{dataset}_dataset", recommendation_dirpart, result["directory"]
            )
            snapshot_dir = f"{model_dir}__topk{value}"
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            shutil.copytree(model_dir, snapshot_dir)

            print(f"Computing metrics for {dataset}/{result['model']} @ top_k_resample={value}")
            df = compute_metrics_for_run(dataset, result["model"], snapshot_dir)
            df["top_k_resample"] = value
            all_rows.append(df)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values", type=int, nargs="+", default=[30, 50, 100, 150])
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "top_k_resample_sweep.csv")

    all_results = []
    for value in args.values:
        if value > 150:
            print(f"Skipping {value}: exceeds recbole_full_casestudy.py's hardcoded k=150 "
                  f"raw pool (bump k there and rerun RecBole first)")
            continue
        for stage in STAGES:
            run_stage(stage, value)
        all_results.append(snapshot_and_measure(value))

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f"\nWrote {len(combined)} rows to {out_path}")

    summary = combined.groupby(["top_k_resample", "method", "metric"])["value"].mean().unstack("metric")
    print("\nMean per method per top_k_resample:")
    print(summary)


if __name__ == "__main__":
    main()
