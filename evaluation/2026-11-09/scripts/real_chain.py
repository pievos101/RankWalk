import os,time,subprocess
from pathlib import Path
R=Path('/workspace/rankwalk-paper-reproduction');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',TF_CPP_MIN_LOG_LEVEL='2')
for rep in range(1,11):
 for cohort in ['PROTHRO','RENAL','PBC2','AIDS','HEART','PAQUID']:
  path=R/'data'/f'clinical_{cohort}_long.csv'
  if not path.exists():
   subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/export_clinical.R')],env=env,check=True)
  if not path.exists():continue
  for method in ['fPCA','RankWalk','TS2Vec','TimesURL','VaDER']:
   out=R/'results'/f'extension_{cohort}_{method}_{rep:02d}.json'
   if out.exists():continue
   python='/workspace/rankwalk-research/vader-venv/bin/python' if method=='VaDER' else '/workspace/.rankwalk-venv/bin/python'
   with (R/'logs'/f'extension_{cohort}_{method}_{rep:02d}.log').open('w') as f:
    p=subprocess.run([python,str(R/'scripts/real_one.py'),cohort,method,str(rep)],env=env,stdout=f,stderr=subprocess.STDOUT)
   print(cohort,method,rep,p.returncode,flush=True)
