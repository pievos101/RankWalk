import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TF_CPP_MIN_LOG_LEVEL='2',PYTHONDONTWRITEBYTECODE='1')
import sys,time,json,random,warnings
from pathlib import Path
import numpy as np,tensorflow as tf
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0,'/workspace/rankwalk-research/VaDER');from VaDER.vader import VADER
path=Path(sys.argv[1]);R=Path('/workspace/rankwalk-paper-reproduction');key=path.stem.removesuffix('_vader');i=int(key.rsplit('_',1)[1]);seed=10000+i
out=R/'results'/f'{key}_VaDER.json'
if out.exists():sys.exit(0)
tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
np.random.seed(seed);random.seed(seed);tf.random.set_seed(seed);warnings.filterwarnings('error',message='Failed to initialize VaDER with Gaussian mixture')
d=np.load(path);x=d['x'];w=np.ones_like(x);start=time.perf_counter()
model=VADER(X_train=x,W_train=w,y_train=None,n_hidden=[12,2],k=4,learning_rate=1e-3,output_activation=None,recurrent=True,cell_type='LSTM',batch_size=64,seed=seed,n_thread=1)
model.pre_fit(n_epoch=50,verbose=False);model.fit(n_epoch=100,verbose=False);pred=np.asarray(model.cluster(x,w))
assert model.n_epoch==150 and model.y is None and np.isfinite(model.loss).all()
np.savez_compressed(R/'results'/f'{key}_VaDER_embedding.npz',embedding=np.asarray(model.model((x,w))[6]),clusters=pred)
result=dict(scenario=key.rsplit('_',1)[0],replicate=i,seed=seed,method='VaDER',ARI=adjusted_rand_score(d['y'],pred),seconds=time.perf_counter()-start,pretrain_epochs=50,train_epochs=100,implementation='official TensorFlow VaDER',readout='native mixture',normalization='repository R scale, across subjects separately per time and marker')
out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
