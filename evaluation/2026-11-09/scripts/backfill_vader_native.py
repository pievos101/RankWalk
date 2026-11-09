import json,subprocess
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/workspace/rankwalk-paper-reproduction')
for path in (R/'results').glob('extension_*_VaDER_??.json'):
 d=json.loads(path.read_text());stem=path.with_suffix('');native=Path(str(stem)+'_native_clusters.npy');outstem=R/'results'/f"extension_{d['cohort']}_VaDER_native_{d['replicate']:02d}";out=outstem.with_suffix('.json')
 if out.exists() or not native.exists():continue
 ids=pd.read_csv(str(stem)+'_features.csv').subject;features=Path(str(outstem)+'_features.csv');pd.DataFrame({'subject':ids,'cluster':np.load(native)+1}).to_csv(features,index=False);k=2 if d['cohort']=='HEART' else 3
 subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts/evaluate_clinical.R'),str(features),str(R/'data'/f"clinical_{d['cohort']}_survival.csv"),str(outstem),str(d['seed']),str(k),'labels'],check=True)
 d.update(method='VaDER_native',readout='native Gaussian mixture assignments',metrics=pd.read_csv(str(outstem)+'_metrics.csv').to_dict('records'));out.write_text(json.dumps(d,indent=2))
