from pathlib import Path
import re,json
root=Path('/workspace/rankwalk-paper-reproduction');repo=Path('/workspace/RankWalk/experiments')
for cohort in ['PBC2','AIDS','HEART','PAQUID']:
 s=(repo/f'application_Irregular_{cohort}.R').read_text()
 s=s.replace('path.expand("~/rankwalk-venv/bin/python")','"/workspace/.rankwalk-venv/bin/python"')
 s=re.sub(r'library\((joineR|JMbayes2|aricode|survcomp)\)','',s)
 s=s.replace('load("pbc2.RData")','load("/workspace/RankWalk/experiments/pbc2.RData")').replace('load("paquid.rda")','load("/workspace/RankWalk/experiments/paquid.rda")')
 s=s.replace('data(heart.valve)','load("/workspace/rankwalk-research/data_sources/joineR/data/heart.valve.rda")').replace('data(aids)','load("/workspace/rankwalk-research/data_sources/JM/data/aids.rda")')
 # Inject overrides after the embedded Python definition; preserve scientific wrapper settings.
 end=s.index('\n")',s.index('py_run_string('))+4
 patch='''\npy_run_string("\nfrom rankwalk import train_gnn as _original_train, compute_jaccard_sparse\nimport time\ntorch.set_num_threads(2)\ndef compute_jaccard_fast(edge_index, num_nodes, device=None):\n    return compute_jaccard_sparse(edge_index, num_nodes)\ndef train_gnn(*args, **kwargs):\n    kwargs.update(backend='numba', seed=int(EXPERIMENT_SEED), verbose=False)\n    return _original_train(*args, **kwargs)\n")\n'''
 s=s[:end]+patch+s[end:]
 # PAQUID source omitted libraries it uses.
 s='library(reticulate);library(fdapace);library(Hmisc);Sys.setenv(RGL_USE_NULL="TRUE")\n'+s
 loop=re.search(r'for\s*\(\s*(ii|iter)\s+in\s+1:n_iter\s*\)\s*\{',s);v=loop.group(1)
 seed=f'\nset.seed(20000+{v}); py$EXPERIMENT_SEED <- as.integer(20000+{v})\n'
 s=s[:loop.end()]+seed+s[loop.end():]
 # Original PAQUID seed overrides R seed, preserve it.
 checkpoint=f'''\nwrite.csv(RES, "{root}/results/clinical_{cohort}.csv",row.names=FALSE)\nsaveRDS(list(iteration={v},FPCA=X_fpca,RankWalk=feat,long=Longdat2,survival=if(exists("surv_data")) surv_data else surv_df, results=RES),sprintf("{root}/results/clinical_{cohort}_%02d.rds",{v}))\n'''
 s=s.replace('print(RES)', 'print(RES)'+checkpoint,1)
 # Strip final plots; loop closes immediately after the first print/checkpoint.
 pos=s.index(checkpoint)+len(checkpoint);closing=s.index('}',pos);s=s[:closing+1]+'\n'
 # Constant padded columns have undefined scale. A zero standardized column is the explicit compatibility correction.
 s=re.sub(r'(feat\s*<-\s*scale\(feat\))',r'\1\nfeat[!is.finite(feat)]<-0',s)
 (root/'scripts'/f'clinical_{cohort}.R').write_text(s)
