#!/bin/bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
source venv/bin/activate

python3 postprocess_baseline_top_k.py
python3 provider_reranker.py
python3 civic_reranker.py
python3 platform_reranker.py
python3 social_choice_aggregation.py
python3 multi_objective_reranker.py
python3 rrf_aggregation.py
