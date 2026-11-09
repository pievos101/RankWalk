import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,json,time,random
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0,'/workspace/rankwalk-research/ts2vec');from ts2vec import TS2Vec
sys.path.insert(0,str(Path(__file__).parent));from run_rankwalk import cluster
R=Path('/workspace/rankwalk-paper-reproduction');torch.set_num_threads(1)
for i in range(1,51):
 stem=R/'results'/f'S2_regular_{i:02d}_TS2Vec_repo_latent';out=Path(str(stem)+'.json')
 if out.exists():continue
 seed=10000+i;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);d=pd.read_csv(R/'data'/f'S2_regular_{i:02d}.csv').sort_values(['subject','time'],kind='stable');ids=sorted(d.subject.unique());cols=[c for c in d.columns if c not in ['subject','time','cluster']];x=np.stack([d[d.subject==s][cols].to_numpy() for s in ids]).astype('float32');y=d.groupby('subject').cluster.first().sort_index().to_numpy();start=time.perf_counter()
 model=TS2Vec(input_dims=x.shape[2],device='cpu',output_dims=64);model.fit(x,n_epochs=50,verbose=False);e=model.encode(x,encoding_window='full_series');pred=cluster(e,stem,seed,nstart=25)
 result=dict(scenario='S2_regular_repo_latent_DIAGNOSTIC',replicate=i,method='TS2Vec_repo',seed=seed,ARI=adjusted_rand_score(y,pred),seconds=time.perf_counter()-start,input_shape=list(x.shape),status='Leaked latent regime input; source reproduction diagnostic, NOT a valid observed-only competitor')
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
