import os,time,subprocess,json
from pathlib import Path
R=Path('/workspace/rankwalk-paper-reproduction');env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
while True:
 for script in ['export_clinical.R','clinical_sensitivity.R','paquid_agedem.R']:
  with (R/'logs'/(script+'.log')).open('a') as f:subprocess.run([str(R/'r-env/bin/Rscript'),str(R/'scripts'/script)],env=env,stdout=f,stderr=subprocess.STDOUT)
 with (R/'logs/report.log').open('w') as f:subprocess.run(['/workspace/.rankwalk-venv/bin/python',str(R/'scripts/build_report.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
 time.sleep(120)
