#!/bin/bash
#SBATCH --job-name="rerank_and_aggregate"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/slurm-out/%x-%j.out
#SBATCH --error=/home/%u/slurm-out/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

./run_rerank_and_aggregate.sh
