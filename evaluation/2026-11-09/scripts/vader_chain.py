import subprocess,time,os
from pathlib import Path
R=Path('/workspace/rankwalk-paper-reproduction');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
for i in range(1,51):
 for s in ['S1_clean','S1_noisy','S2_regular']:
  path=R/'data'/f'{s}_{i:02d}_vader.npz'
  while not path.exists():time.sleep(5)
  with (R/'logs'/f'{s}_{i:02d}_VaDER.log').open('w') as f:
   result=subprocess.run(['/workspace/rankwalk-research/vader-venv/bin/python',str(R/'scripts/vader_one.py'),str(path)],env=env,stdout=f,stderr=subprocess.STDOUT)
  print(s,i,result.returncode,flush=True)
