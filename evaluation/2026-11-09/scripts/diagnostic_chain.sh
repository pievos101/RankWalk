#!/bin/bash
set -eu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/workspace/rankwalk-paper-reproduction
while kill -0 29581 2>/dev/null; do sleep 20; done
"$root/r-env/bin/Rscript" "$root/scripts/repository_diagnostics.R" > "$root/logs/repository_diagnostics.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/ts2vec_latent_diagnostic.py" > "$root/logs/ts2vec_latent_diagnostic.log" 2>&1
