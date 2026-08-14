#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate

python3 ./preprocessing/data_sampling.py
python3 ./preprocessing/metadata_dataset_sample.py

datasets=$(python3 -c "from globals import available_datasets; print(' '.join(available_datasets))")
for dataset in $datasets; do
    cp -r "datasets/${dataset}_dataset/processed_data_recbole" "recbole_general_recs/dataset/${dataset}_sample"
done

