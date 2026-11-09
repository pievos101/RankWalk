from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import hashlib,shutil
R=Path('/workspace/rankwalk-paper-reproduction');O=R/'latex-report';target=R/'RankWalk_LaTeX_Report.zip'
include=[]
for p in O.rglob('*'):
 if not p.is_file():continue
 rel=p.relative_to(O)
 if len(rel.parts)==1 and p.name not in ['report.tex','report.pdf','README.md','Makefile']:continue
 include.append(p)
checks='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(O)) for p in sorted(include))+'\n';(O/'SHA256SUMS').write_text(checks);include.append(O/'SHA256SUMS')
with ZipFile(target,'w',ZIP_DEFLATED) as z:
 for p in include:z.write(p,Path('RankWalk_LaTeX_Report')/p.relative_to(O))
shutil.copy2(O/'report.pdf',R/'RankWalk_Evaluation_Report.pdf')
with ZipFile(target) as z:
 assert z.testzip() is None
 print('ZIP files:',len(z.namelist()),'plots:',len([n for n in z.namelist() if '/figures/' in n]),'bytes:',target.stat().st_size)
 z.extractall('/tmp/rankwalk-latex-portability-check')
print(target)
