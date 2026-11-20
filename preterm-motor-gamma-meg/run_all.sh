#!/usr/bin/env bash
# Run the full analysis in order. Requires a configuration file (see config_example.py and README.md).
set -euo pipefail
cd "$(dirname "$0")/code"
for s in 01_extract_meg_measures.py 02_build_dataset.py 03_participants_behaviour.py 04_gamma_analyses.py \
         05_correlations.py 06_sensitivity_analyses.py 07_beta.py 08_freesurfer.py 09_figures.py 10_power.py; do
  echo "== $s"
  python3 "$s"
done
