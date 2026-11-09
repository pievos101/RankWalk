import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
import sys,time,json,subprocess
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import adjusted_rand_score
from rankwalk import build_temporal_graph,build_temporal_graph_grid,compute_jaccard_sparse,train_gnn
R=Path('/workspace/rankwalk-paper-reproduction');torch.set_num_threads(2)
def cluster(x,stem,seed,k=4,nstart=50):
    pd.DataFrame(x).to_csv(str(stem)+'_features.csv',index=False)
    subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/cluster.R'),str(stem)+'_features.csv',str(stem)+'_clusters.csv',str(seed),'scale',str(k),str(nstart)],check=True,stdout=subprocess.DEVNULL)
    return pd.read_csv(str(stem)+'_clusters.csv').cluster.to_numpy()
def run(path):
    stem=R/'results'/path.stem;out=Path(str(stem)+'_RankWalk.json')
    if out.exists():return
    seed=10000+int(path.stem.rsplit('_',1)[1]);np.random.seed(seed);torch.manual_seed(seed)
    df=pd.read_csv(path);start=time.perf_counter();irregular='irregular' in path.stem
    g,_=(build_temporal_graph_grid(df,n_bins=5,k_similarity=5,overlap=.5) if irregular else build_temporal_graph(df,k_similarity=5))
    nodes=list(g);x=torch.tensor(np.array([g.nodes[n]['features'] for n in nodes]),dtype=torch.float32)
    if not irregular:
        t=torch.tensor([g.nodes[n]['time'] for n in nodes],dtype=torch.float32).reshape(-1,1);x=torch.cat([x,(t-t.mean())/(t.std()+1e-8)],1)
    edges=[];types=[]
    for u,v,a in g.edges(data=True):edges.extend([[u,v],[v,u]]);types.extend([a['edge_type']]*2)
    ei=torch.tensor(edges,dtype=torch.long).T.contiguous();et=torch.tensor(types,dtype=torch.long)
    j=compute_jaccard_sparse(ei,len(nodes));graph_sec=time.perf_counter()-start
    emb=train_gnn(x,ei,et,j,epochs=100,walk_length=20,top_k=10,backend='numba',seed=seed,verbose=False).detach().numpy();train_sec=time.perf_counter()-start-graph_sec
    sub=np.array([g.nodes[n]['subject'] for n in nodes]);t=np.array([g.nodes[n]['time'] for n in nodes]);ids=sorted(set(sub));features=[]
    for s in ids:
        ix=np.where(sub==s)[0];ix=ix[np.argsort(t[ix])];e=emb[ix]
        if irregular:
            grid=np.linspace(df.time.min(),df.time.max(),10)
            e=np.stack([np.interp(grid,t[ix],e[:,c]) for c in range(e.shape[1])],1)
        features.append(e.reshape(-1))
    features=np.stack(features);pred=cluster(features,Path(str(stem)+'_RankWalk'),seed)
    y=df.groupby('subject').cluster.first().sort_index().to_numpy()
    np.savez_compressed(str(stem)+'_RankWalk_embedding.npz',embedding=emb,subjects=sub,time=t,features=features)
    result=dict(scenario=path.stem.rsplit('_',1)[0],replicate=int(path.stem.rsplit('_',1)[1]),method='RankWalk',seed=seed,ARI=adjusted_rand_score(y,pred),seconds=time.perf_counter()-start,graph_seconds=graph_sec,training_seconds=train_sec,subjects=len(ids),nodes=len(nodes),readout='interpolate10-global-grid-flatten' if irregular else 'repository-flatten',backend='numba',epochs=100)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
    scenarios=sys.argv[1:] or ['S1_clean','S1_noisy','S2_regular','S1_irregular','S2_irregular','S1_subject_curve']
    for i in range(1,51):
        for scenario in scenarios:
            path=R/'data'/f'{scenario}_{i:02d}.csv'
            while not path.exists():time.sleep(2)
            run(path)
