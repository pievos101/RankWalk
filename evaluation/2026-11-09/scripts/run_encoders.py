import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,subprocess,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0,'/workspace/rankwalk-research/ts2vec');from ts2vec import TS2Vec
sys.path.insert(0,str(Path(__file__).parent));from run_rankwalk import cluster
R=Path('/workspace/rankwalk-paper-reproduction');torch.set_num_threads(1)
for i in range(1,51):
 for scenario in ['S1_clean','S1_noisy','S2_regular']:
  path=R/'data'/f'{scenario}_{i:02d}.csv'
  while not path.exists():time.sleep(2)
  df=pd.read_csv(path).sort_values(['subject','time'],kind='stable');ids=sorted(df.subject.unique());seed=10000+i;y=df.groupby('subject').cluster.first().sort_index().to_numpy()
  wide=df.pivot_table(index=['subject','time'],columns='outcome',values='y').sort_index();X=np.stack([wide.loc[s].to_numpy() for s in ids]).astype('float32')
  # R VaDER wrapper scales each time/marker across subjects, using sample SD.
  v=(X-X.mean(0,keepdims=True))/(X.std(0,ddof=1,keepdims=True)+1e-12)
  np.savez_compressed(R/'data'/f'{scenario}_{i:02d}_vader.npz',x=v,y=y)
  # Observed-wide is the valid baseline; repository serialization retained separately.
  for method in ['TS2Vec_observed','TS2Vec_repo']:
   stem=R/'results'/f'{scenario}_{i:02d}_{method}';out=Path(str(stem)+'.json')
   if out.exists():continue
   np.random.seed(seed);torch.manual_seed(seed);random.seed(seed);start=time.perf_counter()
   cols=[c for c in df.columns if c not in ['subject','time','cluster']]
   if scenario=='S2_regular' and method=='TS2Vec_repo':
    # Do not train a comparator with simulator latent state as an input.
    out.write_text(json.dumps(dict(scenario=scenario,replicate=i,method=method,status='excluded: repository input includes latent regime; not an observed-data comparator')));continue
   x=X if method=='TS2Vec_observed' else np.stack([df[df.subject==s][cols].to_numpy() for s in ids]).astype('float32')
   model=TS2Vec(input_dims=x.shape[2],device='cpu',output_dims=64)
   model.fit(x,n_epochs=50,verbose=False);e=model.encode(x,encoding_window='full_series')
   pred=cluster(e,stem,seed,nstart=25);result=dict(scenario=scenario,replicate=i,seed=seed,method=method,ARI=adjusted_rand_score(y,pred),seconds=time.perf_counter()-start,epochs=50,input_shape=list(x.shape),implementation='official TS2Vec',output_dims=64)
   np.savez_compressed(str(stem)+'_embedding.npz',embedding=e,subjects=ids)
   out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
