import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import time,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/workspace/rankwalk-paper-reproduction');cohorts=['PROTHRO','RENAL','PBC2','AIDS','HEART','PAQUID'];methods=['TimeOnly','ValueSummary']
while True:
 missing=False
 for cohort in cohorts:
  p=R/'data'/f'clinical_{cohort}_long.csv'
  if not p.exists():missing=True;continue
  d=pd.read_csv(p);survival=pd.read_csv(R/'data'/f'clinical_{cohort}_survival.csv');ids=sorted(set(d.subject)&set(survival.id));d=d[d.subject.isin(ids)];markers=sorted(d.outcome.unique());features={m:[] for m in methods}
  for sid in ids:
   sub=d[d.subject==sid];times=np.sort(sub.time.unique());features['TimeOnly'].append([times.min(),times.max(),times.mean(),times.std(),len(times),np.ptp(times)])
   values=[]
   for marker in markers:
    z=sub[sub.outcome==marker].groupby('time').y.mean().sort_index();v=z.values;t=z.index.to_numpy();centered=t-t.mean() if len(t) else np.array([])
    if not len(v):v=np.array([d[d.outcome==marker].y.mean()]);centered=np.zeros(1)
    slope=float(centered@(v-v.mean())/(centered@centered)) if centered@centered>0 else 0.
    values.extend([v.mean(),v.std(),v[0],v[-1],slope])
   features['ValueSummary'].append(values)
  for method in methods:
   fp=R/'results'/f'control_{cohort}_{method}_features.csv';pd.DataFrame(features[method],index=ids).rename_axis('subject').reset_index().to_csv(fp,index=False)
   for rep in range(1,11):
    stem=R/'results'/f'extension_{cohort}_{method}_{rep:02d}';out=stem.with_suffix('.json')
    if out.exists():continue
    seed=30000+rep;start=time.perf_counter();k=2 if cohort=='HEART' else 3
    subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/evaluate_clinical.R'),str(fp),str(R/'data'/f'clinical_{cohort}_survival.csv'),str(stem),str(seed),str(k)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    result=dict(cohort=cohort,method=method,replicate=rep,seed=seed,subjects=len(ids),seconds=time.perf_counter()-start,scope='descriptive follow-up association control; not held-out prediction',readout='standardized features, KMeans100',metrics=pd.read_csv(str(stem)+'_metrics.csv').to_dict('records'));out.write_text(json.dumps(result,indent=2));print(cohort,method,rep,flush=True)
 if not missing:break
 time.sleep(30)
