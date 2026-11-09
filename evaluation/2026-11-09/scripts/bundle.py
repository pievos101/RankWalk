from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import fcntl
R=Path('/workspace/rankwalk-paper-reproduction');lock=(R/'report.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
temp=R/'RankWalk_Paper_Reproduction.building.zip'
with ZipFile(temp,'w',ZIP_DEFLATED) as z:
 for folder in ['report','scripts']:
  for p in (R/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in str(p) and '.building.' not in p.name:z.write(p,p.relative_to(R))
 for p in (R/'results').glob('*.json'):z.write(p,p.relative_to(R))
 for p in R.glob('*_working_tree.patch'):z.write(p,p.name)
 for name in ['README.md','failed_runs.json','manifest.json','progress_validation.json','python-freeze.txt','vader-freeze.txt','R-session.txt']:
  p=R/name
  if p.exists():z.write(p,name)
temp.replace(R/'RankWalk_Paper_Reproduction.zip')
