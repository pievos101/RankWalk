#!/bin/bash
set -eu
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/workspace/rankwalk-paper-reproduction
# Wait for the already running PBC primary replication to finish.
while ! "$root/r-env/bin/Rscript" -e 'x<-try(read.csv("/workspace/rankwalk-paper-reproduction/results/clinical_PBC2.csv"),silent=TRUE);quit(status=if(inherits(x,"try-error")||anyNA(x)) 1 else 0)' >/dev/null 2>&1; do sleep 10; done
for cohort in AIDS HEART PAQUID; do
 "$root/r-env/bin/Rscript" "$root/scripts/clinical_$cohort.R" > "$root/logs/clinical_$cohort.log" 2>&1 || echo "FAILED $cohort"
 "$root/r-env/bin/Rscript" "$root/scripts/export_clinical.R"
done
