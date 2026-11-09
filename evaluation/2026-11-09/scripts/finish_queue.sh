#!/bin/bash
set -u
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
root=/workspace/rankwalk-paper-reproduction
for pid in 27843 29581 29664 29932 31185; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
# All first-pass workers have stopped; checkpoint-aware retry fills missing jobs.
/workspace/.rankwalk-venv/bin/python "$root/scripts/run_rankwalk.py" > "$root/logs/rankwalk_retry.log" 2>&1
"$root/r-env/bin/Rscript" "$root/scripts/run_baselines.R" > "$root/logs/baselines_retry.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/run_encoders.py" > "$root/logs/encoders_retry.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/vader_chain.py" > "$root/logs/vader_retry.log" 2>&1
for cohort in PBC2 AIDS HEART PAQUID; do "$root/r-env/bin/Rscript" "$root/scripts/clinical_$cohort.R" > "$root/logs/clinical_${cohort}_retry.log" 2>&1; done
"$root/r-env/bin/Rscript" "$root/scripts/export_clinical.R"
/workspace/.rankwalk-venv/bin/python "$root/scripts/real_chain.py" > "$root/logs/real_retry.log" 2>&1
"$root/r-env/bin/Rscript" "$root/scripts/repository_diagnostics.R" > "$root/logs/diagnostics_retry.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/ts2vec_latent_diagnostic.py" > "$root/logs/ts2vec_diagnostic_retry.log" 2>&1
"$root/r-env/bin/Rscript" "$root/scripts/clinical_sensitivity.R" > "$root/logs/sensitivity_retry.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/validate_progress.py"
/workspace/.rankwalk-venv/bin/python "$root/scripts/build_report.py" > "$root/logs/final_report.log" 2>&1
/workspace/.rankwalk-venv/bin/python "$root/scripts/freeze_manifest.py"
date -u > "$root/queue_finished.txt"
/workspace/.rankwalk-venv/bin/python "$root/scripts/bundle.py"
