import json,subprocess
from pathlib import Path
import pandas as pd,numpy as np
R=Path('/workspace/rankwalk-paper-reproduction');problems=[];counts={};missing=[]
for s in ['S1_clean','S1_noisy','S2_regular','S1_irregular','S2_irregular','S1_subject_curve']:
 methods=['RankWalk'] if 'irregular' in s else ['RankWalk','fPCA'] if s=='S1_subject_curve' else ['RankWalk','fPCA','DTW','TAPIO_PC1','TAPIO_weighted','kml3d','TS2Vec_observed','VaDER']
 for method in methods:
  n=0
  for i in range(1,51):
   base=R/'results'/f'{s}_{i:02d}_{method}';j=base.with_suffix('.json');c=base.with_suffix('.csv')
   if j.exists():d=json.loads(j.read_text());value=d.get('ARI');
   elif c.exists():value=float(pd.read_csv(c).ARI.iloc[0])
   else:missing.append(str(base.name));continue
   if value is None or not np.isfinite(value) or value < -1 or value > 1:problems.append(str(base.name))
   else:n+=1
  counts[f'{s}/{method}']=n
for cohort,target in [('PBC2',20),('AIDS',20),('HEART',20),('PAQUID',30)]:
 p=R/'results'/f'clinical_{cohort}.csv';n=len(pd.read_csv(p).dropna()) if p.exists() else 0;counts[f'clinical/{cohort}']=n
 if n<target:missing.append(f'{cohort}: {target-n} primary clinical fits')
 for method in ['fPCA','RankWalk','TS2Vec','TimesURL','VaDER','TimeOnly','ValueSummary']:
  n=sum(json.loads(p.read_text()).get('method')==method for p in (R/'results').glob(f'extension_{cohort}_{method}_*.json'));counts[f'extension/{cohort}/{method}']=n
  if n<10:missing.append(f'{cohort}/{method}: {10-n} extension fits')
for cohort in ['PROTHRO','RENAL']:
 for method in ['fPCA','RankWalk','TS2Vec','TimesURL','VaDER','TimeOnly','ValueSummary']:
  n=sum(json.loads(p.read_text()).get('method')==method for p in (R/'results').glob(f'extension_{cohort}_{method}_*.json'));counts[f'extension/{cohort}/{method}']=n
  if n<10:missing.append(f'{cohort}/{method}: {10-n} extension fits')
for scenario in ['S1_clean','S1_noisy']:
 n=sum((R/'results'/f'{scenario}_{i:02d}_TS2Vec_repo.json').exists() for i in range(1,51));counts[scenario+'/TS2Vec_repo']=n
 if n<50:missing.append(f'{scenario}/TS2Vec_repo: {50-n} runs')
for method in ['fPCA','DTW','TAPIO_PC1','TAPIO_weighted','kml3d','TS2Vec_repo']:
 n=sum((R/'results'/f'S2_regular_{i:02d}_{method+"_repo_latent" if method!="TS2Vec_repo" else "TS2Vec_repo_latent"}.{ "csv" if method!="TS2Vec_repo" else "json"}').exists() for i in range(1,51));counts['latent diagnostic/'+method]=n
 if n<50:missing.append(f'latent diagnostic/{method}: {50-n} runs')
for cohort,target in [('PBC2',20),('AIDS',20),('HEART',20),('PAQUID',30)]:
 n=len(list((R/'results').glob(f'sensitivity_{cohort}_??.csv')));counts['sensitivity/'+cohort]=n
 if n<target:missing.append(f'{cohort}: {target-n} K4 sensitivities')
n=len(list((R/'results').glob('agedem_PAQUID_??.csv')));counts['PAQUID onset correction']=n
if n<30:missing.append(f'PAQUID: {30-n} onset-corrected sensitivities')
for cohort in ['PROTHRO','RENAL','PBC2','AIDS','HEART','PAQUID']:
 n=len(list((R/'results').glob(f'extension_{cohort}_VaDER_native_??.json')));counts['native VaDER/'+cohort]=n
 if n<10:missing.append(f'{cohort}: {10-n} native VaDER readouts')
repo_status=subprocess.check_output(['git','-C','/workspace/RankWalk','status','--porcelain'],text=True)
result=dict(queue_finished=(R/'queue_finished.txt').exists(),failure_audit='failed_runs.json',complete=not missing and not problems,counts=counts,missing=missing,invalid_results=problems,repository_status=repo_status,no_push_policy='All experimental scripts and artifacts outside repository; no push commands used')
(R/'progress_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(complete=result['complete'],invalid_results=problems,completed=sum(counts.values()),missing_jobs=len(missing),repository_clean=not repo_status)))
