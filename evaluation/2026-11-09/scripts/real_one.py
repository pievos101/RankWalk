import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,random,subprocess
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/workspace/rankwalk-paper-reproduction');cohort,method,rep=sys.argv[1],sys.argv[2],int(sys.argv[3]);seed=30000+rep;stem=R/'results'/f'extension_{cohort}_{method}_{rep:02d}';out=Path(str(stem)+'.json')
if out.exists():sys.exit()
start=time.perf_counter();np.random.seed(seed);random.seed(seed)
d=pd.read_csv(R/'data'/f'clinical_{cohort}_long.csv');surv=pd.read_csv(R/'data'/f'clinical_{cohort}_survival.csv');ids=sorted(set(d.subject)&set(surv.id));d=d[d.subject.isin(ids)];markers=sorted(d.outcome.unique());k=2 if cohort=='HEART' else 3
X=[];grid=np.linspace(d.time.min(),d.time.max(),32)
for s in ids:
 series=[]
 for m in markers:
  z=d[(d.subject==s)&(d.outcome==m)].groupby('time').y.mean().sort_index();series.append(np.interp(grid,z.index,z.values) if len(z) else np.full(32,d[d.outcome==m].y.mean()))
 X.append(np.stack(series,axis=1))
X=np.array(X,dtype='float32');X=(X-X.mean((0,1),keepdims=True))/(X.std((0,1),keepdims=True)+1e-6)
if method=='fPCA':
 features=R/'results'/f'extension_{cohort}_fPCA_features.csv'
 if not features.exists():subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/clinical_fpca.R'),str(R/'data'/f'clinical_{cohort}_long.csv'),str(features)],check=True)
 e=pd.read_csv(features).set_index('subject').loc[ids].to_numpy()
elif method=='VaDER':
 import tensorflow as tf,warnings
 tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1);tf.random.set_seed(seed)
 warnings.filterwarnings('error',message='Failed to initialize VaDER with Gaussian mixture')
 sys.path.insert(0,'/workspace/rankwalk-research/VaDER');from VaDER.vader import VADER
 model=VADER(X_train=X,W_train=np.ones_like(X),y_train=None,n_hidden=[12,2],k=k,learning_rate=.001,output_activation=None,recurrent=True,cell_type='LSTM',batch_size=64,seed=seed,n_thread=1)
 model.pre_fit(n_epoch=50,verbose=False);model.fit(n_epoch=100,verbose=False);e=np.asarray(model.model((X,np.ones_like(X)))[6]);native=np.asarray(model.cluster(X,np.ones_like(X)));np.save(str(stem)+'_native_clusters.npy',native)
else:
 import torch
 torch.set_num_threads(1);torch.manual_seed(seed)
 if method=='TS2Vec':
  sys.path.insert(0,'/workspace/rankwalk-research/ts2vec');from ts2vec import TS2Vec
  model=TS2Vec(input_dims=X.shape[-1],device='cpu',output_dims=64);model.fit(X,n_epochs=50,verbose=False);e=model.encode(X,encoding_window='full_series')
 elif method=='TimesURL':
  from types import SimpleNamespace
  sys.path.insert(0,'/workspace/rankwalk-research/TimesURL/src');from timesurl import TimesURL
  t=np.broadcast_to(np.linspace(0,1,32,dtype='float32')[None,:,None],(len(X),32,1));data={'x':np.concatenate([X,t],axis=-1),'mask':np.ones_like(X)}
  model=TimesURL(input_dims=X.shape[-1],output_dims=64,hidden_dims=64,depth=10,device='cpu',batch_size=16,lr=.0001,args=SimpleNamespace(batch_size=16,lmd=.01,segment_num=3,mask_ratio_per_seg=.05));model.fit(data,n_iters=200,verbose=False);e=model.encode(data,encoding_window='full_series')
 elif method=='RankWalk':
  from rankwalk import build_temporal_graph_grid,compute_jaccard_sparse,train_gnn
  g,_=build_temporal_graph_grid(d,k_similarity=5,n_bins=5,overlap=.5);nodes=list(g);x=torch.tensor(np.array([g.nodes[n]['features'] for n in nodes]),dtype=torch.float32);t=torch.tensor([g.nodes[n]['time'] for n in nodes],dtype=torch.float32).reshape(-1,1);x=torch.cat([x,(t-t.mean())/(t.std()+1e-8)],1)
  edges=[];types=[]
  for u,v,a in g.edges(data=True):edges.extend([[u,v],[v,u]]);types.extend([a['edge_type']]*2)
  ei=torch.tensor(edges,dtype=torch.long).T.contiguous();et=torch.tensor(types,dtype=torch.long);j=compute_jaccard_sparse(ei,len(nodes));emb=train_gnn(x,ei,et,j,epochs=100,walk_length=20,top_k=10,backend='numba',seed=seed,verbose=False).detach().numpy();subjects=np.array([g.nodes[n]['subject'] for n in nodes]);e=np.stack([emb[subjects==s].mean(0) for s in ids])
 else:raise ValueError(method)
assert len(e)==len(ids) and np.isfinite(e).all()
features=Path(str(stem)+'_features.csv');pd.DataFrame(e,index=ids).rename_axis('subject').reset_index().to_csv(features,index=False)
subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/evaluate_clinical.R'),str(features),str(R/'data'/f'clinical_{cohort}_survival.csv'),str(stem),str(seed),str(k)],check=True)
result=dict(cohort=cohort,method=method,replicate=rep,seed=seed,subjects=len(ids),seconds=time.perf_counter()-start,scope='extension, common observed cohort, full-follow-up descriptive association',readout='standardized embedding + KMeans100, source K and K4 sensitivity',preprocessing='observed graph for RankWalk; native sparse PACE for FPCA; absolute-time 32-grid interpolation plus global marker standardization for encoders',metrics=pd.read_csv(str(stem)+'_metrics.csv').to_dict('records'))
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if method=='VaDER':
 native_stem=R/'results'/f'extension_{cohort}_VaDER_native_{rep:02d}'
 native_features=Path(str(native_stem)+'_features.csv');pd.DataFrame({'subject':ids,'cluster':native+1}).to_csv(native_features,index=False)
 subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/evaluate_clinical.R'),str(native_features),str(R/'data'/f'clinical_{cohort}_survival.csv'),str(native_stem),str(seed),str(k),'labels'],check=True)
 extra=dict(result,method='VaDER_native',readout='native Gaussian mixture labels, no KMeans',metrics=pd.read_csv(str(native_stem)+'_metrics.csv').to_dict('records'))
 Path(str(native_stem)+'.json').write_text(json.dumps(extra,indent=2))
