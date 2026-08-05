#!/bin/bash
#SBATCH --job-name="recbole_hyperopt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/%u/slurm-out/%x-%j.out
#SBATCH --error=/home/%u/slurm-out/%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate

python3 recbole_general_recs/config_hyperparameter_creator.py
python3 recbole_general_recs/recbole_full_casestudy.py
