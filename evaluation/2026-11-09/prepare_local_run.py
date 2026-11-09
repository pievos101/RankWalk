"""Relocate the archived scientific scripts into an isolated local working directory."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
from zipfile import ZipFile

def main():
    study=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,default=study/'work')
    p.add_argument('--python',default=sys.executable,help='Scientific Python environment with RankWalk installed')
    p.add_argument('--vader-python',help='Separate Python 3.9 environment with official VaDER dependencies')
    p.add_argument('--rscript',default=shutil.which('Rscript'),help='Rscript with the recorded scientific packages installed')
    p.add_argument('--fetch-dependencies',action='store_true',help='Clone pinned public reference implementations into the work directory')
    a=p.parse_args()
    if not a.rscript:p.error('--rscript is required when Rscript is not on PATH')
    w=a.work_dir.expanduser().resolve()
    if any(w==z for z in [study,study/'scripts',study/'report',study/'per_run_scores']):p.error('Choose a separate work directory')
    for d in ['scripts','results','data','logs','references','r-env/bin']: (w/d).mkdir(parents=True,exist_ok=True)
    python=str(Path(a.python).expanduser().absolute());vader=str(Path(a.vader_python).expanduser().absolute()) if a.vader_python else python
    rscript=str(Path(a.rscript).expanduser().resolve())
    if not all(Path(x).is_file() for x in [python,vader,rscript]):p.error('Interpreter paths must exist')
    link=w/'r-env/bin/Rscript'
    if link.is_symlink():link.unlink()
    if link.exists():p.error(f'Refusing to replace an existing file: {link}')
    link.symlink_to(rscript)
    mapping=[('/workspace/rankwalk-research/vader-venv/bin/python',vader),('/workspace/.rankwalk-venv/bin/python',python),('/workspace/rankwalk-paper-reproduction',str(w)),('/workspace/RankWalk',str(study.parents[1])),('/workspace/rankwalk-research',str(w/'references'))]
    scientific=['generate.R','run_rankwalk.py','run_baselines.R','run_encoders.py','vader_one.py','vader_chain.py','real_one.py','real_chain.py','run_controls.py','cluster.R','clinical_fpca.R','evaluate_clinical.R','export_clinical.R','clinical_sensitivity.R','paquid_agedem.R','repository_diagnostics.R','ts2vec_latent_diagnostic.py','backfill_vader_native.py','clinical_PBC2.R','clinical_AIDS.R','clinical_HEART.R','clinical_PAQUID.R']
    for name in scientific:
        s=(study/'scripts'/name).read_text()
        for old,new in mapping:s=s.replace(old,new)
        (w/'scripts'/name).write_text(s)
    with ZipFile(study/'benchmark_inputs.zip') as z:
        for member in z.infolist():
            target=(w/member.filename).resolve()
            if not target.is_relative_to(w):raise ValueError('Unsafe archive member')
        z.extractall(w)
    shutil.copytree(study/'reference_sources/TAPIO',w/'TAPIO',dirs_exist_ok=True)
    manifest=json.loads((study/'report/provenance/manifest.json').read_text())
    if a.fetch_dependencies:
        repos={'ts2vec':('TS2Vec','https://github.com/yuezhihan/ts2vec.git'),'VaDER':('VaDER','https://github.com/johanndejong/VaDER.git'),'TimesURL':('TimesURL','https://github.com/Alrash/TimesURL.git'),'data_sources/JM':('JM','https://github.com/cran/JM.git'),'data_sources/joineR':('joineR','https://github.com/cran/joineR.git'),'data_sources/joineRML':('joineRML','https://github.com/cran/joineRML.git')}
        for rel,(key,url) in repos.items():
            dest=w/'references'/rel
            if not dest.exists():
                dest.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone',url,str(dest)],check=True);subprocess.run(['git','-C',str(dest),'checkout','--detach',manifest['sources'][key]],check=True)
            head=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
            if head!=manifest['sources'][key]:raise RuntimeError(f'Unexpected dependency commit: {dest}')
        patch=study/'report/provenance/TimesURL_working_tree.patch'
        if patch.exists():
            reverse=subprocess.run(['git','-C',str(w/'references/TimesURL'),'apply','--reverse','--check',str(patch)],capture_output=True)
            if reverse.returncode:subprocess.run(['git','-C',str(w/'references/TimesURL'),'apply',str(patch)],check=True)
        # The clinical export script originally used a separate joineRML checkout.
        text=(w/'scripts/export_clinical.R').read_text().replace(str(w/'data_sources/joineRML'),str(w/'references/data_sources/joineRML'))
        (w/'scripts/export_clinical.R').write_text(text)
    text=(w/'scripts/export_clinical.R').read_text().replace(str(w/'data_sources/joineRML'),str(w/'references/data_sources/joineRML'))
    (w/'scripts/export_clinical.R').write_text(text)
    config=dict(work_dir=str(w),python=python,vader_python=vader,rscript=rscript,reference_commits=manifest['sources'])
    (w/'local_run.json').write_text(json.dumps(config,indent=2))
    if not a.vader_python:print('VaDER requires a separate compatible Python 3.9 environment; pass --vader-python before running that method.')
    print('Prepared relocated scientific scripts and recorded inputs:',w)
    print('No model fitting was started. See the study README for execution order.')
if __name__=='__main__':main()
