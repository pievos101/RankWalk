import os
os.environ['MPLCONFIGDIR']='/workspace/rankwalk-paper-reproduction/cache/matplotlib'
from pathlib import Path
import shutil,json,datetime
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/workspace/rankwalk-paper-reproduction');O=R/'latex-report';O.mkdir(exist_ok=True)
for name in ['figures','data','provenance','logs']:(O/name).mkdir(exist_ok=True)
for p in (R/'report/figures').glob('*'):
 if p.suffix in ['.pdf','.png']:shutil.copy2(p,O/'figures'/p.name)
for p in (R/'report').glob('*.csv'):shutil.copy2(p,O/'data'/p.name)
for name in ['manifest.json','failed_runs.json','progress_validation.json','python-freeze.txt','vader-freeze.txt','R-session.txt']:
 shutil.copy2(R/name,O/'provenance'/name)
for p in R.glob('*_working_tree.patch'):shutil.copy2(p,O/'provenance'/p.name)
fail=json.loads((R/'failed_runs.json').read_text())
for item in fail['failures']:shutil.copy2(R/item['log'],O/'logs'/Path(item['log']).name)
sim=pd.read_csv(O/'data/simulation_results.csv');summary=pd.read_csv(O/'data/simulation_summary.csv');ext=pd.read_csv(O/'data/clinical_extension_results.csv');primary=pd.read_csv(O/'data/clinical_primary_results.csv');sens=pd.read_csv(O/'data/sensitivity_all.csv');agedem=pd.read_csv(O/'data/agedem_all.csv')
methods=['RankWalk','fPCA','DTW','kml3d','TAPIO_PC1','TAPIO_weighted','TS2Vec_observed','TS2Vec_repo','VaDER']
names={'fPCA':'FPCA','TAPIO_PC1':'TAPIO PC1','TAPIO_weighted':'TAPIO weighted','TS2Vec_observed':'TS2Vec (observed wide)','TS2Vec_repo':'TS2Vec (source serialization)','VaDER_native':'VaDER (native mixture)','TimeOnly':'Visit times/counts','ValueSummary':'Marker summaries','VaDER':'VaDER (posterior + k-means)'}
def esc(s):
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=str(s).replace(a,b)
 return s
def label(m):return esc(names.get(m,m))
def pm(v,dp=3):return f'${np.mean(v):.{dp}f} \\pm {np.std(v,ddof=1):.{dp}f}$' if len(v)>1 else f'${np.mean(v):.{dp}f}$'
parts=[]
def add(s):parts.append(s)
def table(headers,rows,caption,align=None):
 align=align or 'l'+'c'*(len(headers)-1)
 add('\\begin{table}[htbp]\n\\centering\\small\n\\caption{'+caption+'}\n\\begin{tabular}{'+align+'}\\toprule\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+'\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{table}')
def figure(file,caption,width=r'\linewidth'):
 add(r'\begin{figure}[htbp]\centering\includegraphics[width='+width+']{'+file+r'}\caption{'+caption+r'}\end{figure}')
add(r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx,booktabs,longtable,amsmath,array,xcolor}
\usepackage[hidelinks]{hyperref}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\renewcommand{\arraystretch}{1.15}
\title{RankWalk: Paper Reproduction and Expanded Longitudinal Evaluation}
\author{Computational evaluation report}
\date{9 November 2026}
\begin{document}
\maketitle
\begin{abstract}
This report reruns the RankWalk paper experiments using the corrected CPU-accelerated implementation, reproduces four original clinical cohorts, and extends the comparison to six longitudinal cohorts and additional representation methods. Repository settings are the primary specification; paper-specification sensitivities and compatibility corrections are identified separately. All 300 RankWalk simulation fits and all 90 original clinical fits completed. Three VaDER fits failed numerical checks despite retries. RankWalk is competitive but does not dominate all settings: TS2Vec leads the regime-switching simulation, and simple summaries or TimesURL outperform the common RankWalk configuration on PAQUID. Strong visit-time-only controls show that full-follow-up clinical concordance must be interpreted as descriptive association, not prospective prediction.
\end{abstract}
\tableofcontents
\clearpage
\section{Scope, completion, and statistical interpretation}
The reference is the uploaded paper \emph{Contrastive Representation Learning of Longitudinal Disease Trajectories on Temporal Graphs}, arXiv:2607.25609. The software checkout is RankWalk commit \texttt{58a87af4fdcdd243425106622e0e4f46dcc1ba6a}. This evaluation changed no tracked repository files and pushed nothing.

The design comprises six RankWalk simulation settings with 50 repetitions each; 20 PBC2, 20 AIDS, 20 HEART, and 30 PAQUID original-protocol clinical fits; and ten requested fits per extension method and cohort. Numerical failures are reported, not replaced by selected seeds or fabricated scores. The strict all-successful flag in the provenance audit remains false because of the three failed VaDER fits, although the compute queue has finished.

Simulation accuracy is measured by adjusted Rand index (ARI), with known cluster labels used only for evaluation. Clinical outcomes are summarized by concordance and log-rank chi-square after clustering full observed trajectories. Cluster risk ordering uses observed follow-up times, except the native PAQUID script, which uses a fitted categorical Cox model. These are in-sample, post-hoc association measures. They are not held-out predictive accuracy or AUROC.

Every $\pm$ value is the sample standard deviation across repeated simulation datasets or stochastic fits. Clinical repeats reuse the same patients; their SD is algorithmic variation, not patient-sampling uncertainty. A reported SD of zero means identical repeated results, not absence of uncertainty. No exhaustive competitor hyperparameter search was performed.

\section{Methods and fixed budgets}
Simulations use 200 subjects, four balanced clusters, and five outcomes. Replicate seeds are 10001--10050. Compared methods receive the same generated dataset for each scenario and replicate. RankWalk uses sparse Jaccard similarity and the CPU Numba backend: 100 epochs, Adam learning rate $10^{-3}$, walk length 20, top-$k$ 10, and similarity-neighbor count 5. Regular-series node embeddings are flattened per subject, standardized, and clustered with R k-means using 50 starts.

Native R baselines are retained: B-spline FPCA with five basis functions and two components per marker; DTW distances with PAM; kml3d with ten redrawings; and TAPIO with 500 trees, four levels, and PC1 or variance-weighted PCA selection. Official TS2Vec uses 50 epochs and 64 output dimensions. Official VaDER uses hidden/latent dimensions [12,2], 50 pretraining plus 100 fitting epochs, batch size 64, and its native Gaussian-mixture clusters for simulation ARI.

Real-data extensions additionally include TimesURL (200 updates, 64 output dimensions), simple marker summaries, and visit-time/count controls. Sequence encoders receive a common 32-point absolute-time interpolation with global marker standardization; FPCA uses native sparse PACE; RankWalk uses the observed graph, mean pooling, 100 epochs, walk length 20, top-$k$ 10, five similarity neighbors, and grid overlap 0.5. Standardized representations feed k-means with 100 starts. VaDER is shown both with native mixture assignments and with posterior embeddings followed by common k-means; these are different readouts of the same fitted model, not independent fits.

Visit-time controls contain minimum, maximum, mean, SD, range, and count of observed visit times. Marker summaries contain mean, SD, first and last value, and slope per marker. These unsupervised controls require no outcome labels during representation construction. They expose association carried by the observation process or simple trajectory summaries.

The new cohorts are PROTHRO (488 cirrhosis-trial patients) and RENAL (407 renal-transplant patients). RENAL has over 180,000 marker records; data are aggregated into 32 prespecified absolute-time bins, with GFR log-transformed, before the graph/FPCA comparison. Original cohort sources are public JM, joineR, and joineRML package data. Source commits and package versions accompany this report.

\section{Specification differences and explicit corrections}
\begin{enumerate}
\item \textbf{Simulation 1 random curves.} The repository redraws quadratic random-curve coefficients per visit. The paper describes one curve per subject. The repository version is primary; the subject-shared version is a separate sensitivity. Clean noise uses marker-effect SDs of 5. The noisy-marker condition starts at 3 and replaces one marker SD by an integer drawn from 3--20. Thus the clean and noisy panels differ in base noise; the noisy panel is not an isolated addition of noise to the clean panel.
\item \textbf{Irregular readout.} The source irregular script flattens unequal-length trajectories into incompatible rows. The evaluation interpolates node embeddings to a ten-point global absolute-time grid before flattening. Graph and readout settings differ between regular and irregular pipelines, so their difference cannot be attributed solely to sampling irregularity.
\item \textbf{TS2Vec serialization and latent state.} Simulation 1 source input is $N\times50\times2$ (long-format outcome identifier and value), rather than the observed-marker $N\times10\times5$ tensor. Both are reported. Simulation 2 adds latent regime state to the source TS2Vec input and to the R wide matrix used by other baselines. Observed-only inputs are the valid main comparison. Source-input leakage diagnostics are shown separately and cannot support an observed-data superiority claim.
\item \textbf{Clinical cluster counts.} Source scripts use $K=3$ for PBC2, AIDS and PAQUID and $K=2$ for HEART, whereas the paper text describes $K=4$. Source $K$ is primary; $K=4$ is a separate sensitivity. Actual source GNN calls use 100 epochs for PBC2/HEART/PAQUID and 200 for AIDS. Source HEART/PAQUID graph settings use ten neighbors, overlap 0.8, and walk/top-$k$ 5. HEART's common extension configuration therefore differs materially from its primary configuration.
\item \textbf{PAQUID endpoint and readout.} The subject-level eventual dementia indicator is treated as visit-specific in the source, assigning event time zero to dementia cases. That native result is retained only for reproduction. The extension uses \texttt{agedem} minus baseline age, excludes nonpositive times, and applies common Hmisc concordance. Native PAQUID uses zero-padded flattened embeddings; the common extension uses mean pooling. Correcting the endpoint on saved native representations is a different sensitivity from rerunning the common extension.
\item \textbf{Other compatibility details.} Source AIDS duplicates CD4 as two channels; primary reproduction preserves this and the extension removes duplication. Undefined standardized constant padding columns are set to zero. Primary clinical R clustering, Torch initialization and walk sampling are seeded, but the source clinical NumPy graph-subspace RNG was not pre-seeded. Saved representations permit exact metric replay, while graph refits retain that reproducibility limitation.
\end{enumerate}

\section{Simulation results}
\subsection{Primary regular-data comparison}
''')
rows=[]
for m in methods:
 vals=[]
 for scenario in ['S1_clean','S1_noisy','S2_regular']:
  z=sim[(sim.scenario==scenario)&(sim.method==m)].ARI
  vals.append(pm(z) if len(z) else '--')
 rows.append([label(m)]+vals)
table(['Method','S1 clean','S1 noisy marker','S2 switching'],rows,'Mean ARI $\\pm$ SD. All cells use 50 replicates except VaDER S2 (49 successful fits). Source-serialized TS2Vec is a separate input sensitivity; its leaked S2 version is excluded here.')
add(r'''RankWalk has a small clean-setting advantage over FPCA. Under marker corruption it is essentially tied with weighted TAPIO. TS2Vec with observed-marker input leads Simulation 2. These findings do not support universal superiority of RankWalk. The unusually higher noisy-setting mean for some methods is not evidence that adding noise helps: the two panels use different base-noise settings.''')
# paired bootstrap, fixed seed and unadjusted exploratory intervals
bootrows=[]
for scenario,sc in [('S1_clean','S1 clean'),('S1_noisy','S1 noisy'),('S2_regular','S2')]:
 a=sim[(sim.scenario==scenario)&(sim.method=='RankWalk')].set_index('replicate').ARI
 for m in ['fPCA','TAPIO_weighted','TS2Vec_observed']:
  b=sim[(sim.scenario==scenario)&(sim.method==m)].set_index('replicate').ARI;ix=a.index.intersection(b.index);v=(a.loc[ix]-b.loc[ix]).to_numpy();rng=np.random.default_rng(0);bs=rng.choice(v,size=(10000,len(v)),replace=True).mean(1);lo,hi=np.quantile(bs,[.025,.975]);bootrows.append([sc,label(m),f'{v.mean():+.3f}',f'[{lo:+.3f}, {hi:+.3f}]'])
table(['Setting','Comparator','Mean $\\Delta$ ARI','95\\% bootstrap interval'],bootrows,'Paired RankWalk minus comparator differences. Percentile bootstrap over 50 paired replicates, 10,000 resamples; exploratory intervals are not adjusted for multiple comparisons.')
for name,cap in [('S1_clean','Clean Simulation 1.'),('S1_noisy','Simulation 1 with one corrupted marker.'),('S2_regular','Observed-marker Simulation 2; latent regime is excluded from the comparator input.')]:
 figure(name+'.pdf',cap+' Horizontal violin/box/jitter plots and paired dominance heatmaps follow the reference paper style. A heatmap cell is the proportion of non-tied paired replicates in which the column method beats the row method; gray cells are undefined or diagonal. VaDER S2 comparisons use 49 available pairs.')
add(r'\clearpage\subsection{Irregular sampling and subject-shared random curves}')
rows=[]
for s,title in [('S1_clean','S1 regular'),('S1_irregular','S1 irregular'),('S2_regular','S2 regular'),('S2_irregular','S2 irregular'),('S1_subject_curve','S1 subject-shared curve')]:
 z=sim[(sim.scenario==s)&(sim.method=='RankWalk')].ARI;rows.append([title,str(len(z)),pm(z)])
table(['RankWalk setting','Fits','ARI'],rows,'All 300 RankWalk fits: six settings with 50 repetitions each.')
add(r'''The subject-shared-curve sensitivity gives RankWalk $0.782\pm0.081$ and FPCA $0.756\pm0.070$, substantially below their repository-generator results. This is a material specification effect. High irregular-data scores show recovery under the corrected pipeline; they do not isolate the causal effect of sampling irregularity.''')
figure('S1_clean_vs_irregular.pdf','Simulation 1, regular versus irregular RankWalk pipelines.')
figure('S2_regular_vs_irregular.pdf','Simulation 2, regular versus irregular RankWalk pipelines.')
figure('S1_subject_curve.pdf','Subject-shared random-curve sensitivity, compared with FPCA.')
add(r'\clearpage\section{Original clinical reproduction}')
rows=[]
for c in ['PBC2','AIDS','HEART','PAQUID']:
 for m in ['RankWalk','FPCA']:
  z=primary[(primary.cohort==c)&(primary.method==m)];rows.append([c,m,str(len(z)),pm(z.C_index),pm(z.logrank_chisq,2)])
table(['Cohort','Method','Fits','C-index','Log-rank $\\chi^2$'],rows,'Completed primary clinical reproduction. Source cluster counts and readouts are preserved. Native PAQUID has a time-zero endpoint defect and Cox-derived concordance; it is not a valid incident-dementia prediction analysis.')
add(r'''The fresh FPCA PBC2 values ($C=0.7603$, $\chi^2=127.50$), AIDS values ($0.6423$, $53.09$), and HEART values ($0.5693$, $4.81$) match the paper's rounded values. RankWalk means are also close to the reported values. PAQUID FPCA concordance matches the paper's rounded value, but the fresh source-script log-rank statistic is 34.29 versus the paper's 5.06. This discrepancy is retained rather than overwritten or assumed resolved.''')
for c in ['PBC2','AIDS','HEART','PAQUID']:figure(c+'.pdf',c+' primary reproduction. Boxplots and jitter summarize repeated fits, not independent patient samples.')
add(r'\clearpage\section{Expanded real-world comparison}')
add(r'''The following tables report successful fits on the common eligible cohort, using $K=3$ except HEART ($K=2$). Primary tables summarize each method's successful fits: VaDER has nine successes on RENAL and HEART, while the other methods have ten. Replicate-level data permit matched-subset comparisons; unequal-count means are not paired significance tests. Failure rates are reported separately. Zero SDs in deterministic controls indicate stable partitions on the same patients.''')
ems=['RankWalk','fPCA','TS2Vec','TimesURL','VaDER_native','VaDER','ValueSummary','TimeOnly']
for cohorts in [['PBC2','PAQUID','PROTHRO'],['RENAL','AIDS','HEART']]:
 rows=[]
 for m in ems:
  vals=[]
  for c in cohorts:
   k=2 if c=='HEART' else 3;z=ext[(ext.cohort==c)&(ext.method==m)&(ext.k==k)].C_index;vals.append(pm(z)+(r'$^{\dagger}$' if len(z)<10 else ''))
  rows.append([label(m)]+vals)
 table(['Method']+cohorts,rows,'Mean C-index $\\pm$ SD across successful extension fits. $\\dagger$ denotes nine successful fits rather than ten. VaDER posterior and native-mixture rows are two readouts of the same model.')
rows=[]
for c in ['PBC2','PAQUID','PROTHRO','RENAL','AIDS','HEART']:
 k=2 if c=='HEART' else 3;z=ext[(ext.cohort==c)&(ext.method=='RankWalk')&(ext.k==k)];rows.append([c,str(k),str(int(z.n.iloc[0])),str(int(z.events.iloc[0]))])
table(['Cohort','Primary extension $K$','Subjects','Events'],rows,'Common extension-cohort sizes. PAQUID uses the corrected onset-age endpoint.')
add(r'''PBC2 favors RankWalk among these methods. PAQUID favors marker summaries and TimesURL; mean-pooled RankWalk is approximately comparable to FPCA. PROTHRO and RENAL favor RankWalk among biomarker representations but have even stronger visit-time/count controls. AIDS and HEART also show strong observation-process controls. Consequently, these metrics cannot establish biomarker-specific prognostic benefit or future clinical prediction.

HEART illustrates configuration sensitivity: the source-specific graph/walk configuration reproduces concordance near 0.706, whereas the common extension configuration is substantially weaker. PAQUID's extension also changes both readout and endpoint relative to the native source; its difference is not an isolated effect of faster training or endpoint correction.''')
# Generate compact paper-style final extension figures (all methods, all successful fits).
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
for cohort in ['PBC2','PAQUID','PROTHRO','RENAL','AIDS','HEART']:
 k=2 if cohort=='HEART' else 3;fig,axes=plt.subplots(1,2,figsize=(12,4.8));rng=np.random.default_rng(0)
 for ax,metric,title in [(axes[0],'C_index','Concordance'),(axes[1],'logrank_chisq','Log-rank chi-square')]:
  vals=[ext[(ext.cohort==cohort)&(ext.method==m)&(ext.k==k)][metric].dropna().to_numpy() for m in ems];positions=np.arange(len(ems));bp=ax.boxplot(vals,positions=positions,orientation='horizontal',widths=.5,patch_artist=True,showfliers=False)
  for i,(patch,v) in enumerate(zip(bp['boxes'],vals)):
   patch.set_facecolor(plt.get_cmap('Set2')(i%8));ax.scatter(v,i+rng.uniform(-.12,.12,len(v)),color='black',s=9,alpha=.45)
  ax.set_yticks(positions,[names.get(m,m) for m in ems]);ax.set_title(title);ax.grid(axis='x',alpha=.2)
 fig.suptitle(cohort+f' extension, K={k}; 10 requested fits per method');fig.tight_layout();name='extension_'+cohort;fig.savefig(O/'figures'/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/'figures'/(name+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
 figure(name+'.pdf',cohort+' final extension distributions. Nine native VaDER successes are shown for RENAL and HEART. The PAQUID endpoint is corrected; these are full-follow-up associations.')
add(r'\clearpage\section{Clinical sensitivity checks}')
for data,title in [(sens,'Saved-representation cluster-count sensitivity'),(agedem,'PAQUID onset-age correction on saved native representations')]:
 add('\\subsection{'+title+'}')
 rows=[]
 for (c,m,k),z in data.groupby(['cohort','method','k']):rows.append([c,label(m),str(k),str(len(z)),pm(z.C_index),pm(z.logrank_chisq,2)])
 table(['Cohort','Method','$K$','Fits','C-index','Log-rank $\\chi^2$'],rows,title+'. Common Hmisc concordance and 100-start k-means are used; this is not identical to the native PAQUID Cox score.')
add(r'''The onset-corrected analysis of saved native PAQUID flattened representations and the newly trained mean-pooling extension are distinct experiments. Their scores should not be pooled or substituted for each other. Changing $K$ also changes log-rank degrees of freedom; larger raw chi-square alone is not a fair cross-$K$ selection criterion.''')
add(r'\clearpage\section{Failures, runtime, and limitations}')
table(['Setting','VaDER success / attempted','Failure'],[['S2 regular','49 / 50','Singular covariance'],['RENAL extension','9 / 10','Mixture initialization covariance'],['HEART extension','9 / 10','Singular covariance']],'Documented numerical failures. Original seeds and model settings were retained on retry; no successful replacement seed was selected.')
add(r'''Failures remain part of the evaluation. Successful-fit summaries do not assign zero scores to failures and are therefore conditional on success. The failure audit and original error logs are bundled. All RankWalk fits completed.

Runtimes were collected during concurrent work on four CPU cores and under different algorithm-specific budgets. They are not controlled isolated speed comparisons. Sparse Jaccard and the Numba sampler were active for RankWalk; stochastic streams differ from the Torch backend, so identical learned partitions are not guaranteed. No new isolated old-versus-fast speedup factor is claimed here.

Clinical graph construction and representation learning are transductive over the observed cohort. Concordance risk ordering and categorical Cox fitting reuse outcome information for evaluation. Visit duration, density and eventual follow-up contribute substantial signal. A separate fixed-landmark design, appropriate censoring treatment and held-out patient evaluation are needed to assess prospective predictive value. Clinical uncertainty should additionally be assessed by patient resampling, rather than interpreting fit-to-fit SD as a confidence interval.

The competitors are selected established methods, not an exhaustive survey of every method available in 2026. Their fixed source budgets are not equal-FLOP budgets or a fully optimized performance frontier. New cohort selection and model comparisons are exploratory.

\section{Reproducibility package}
This source compiles without absolute workspace paths. Run \texttt{make} or execute \texttt{pdflatex report.tex} twice. The \texttt{figures/} directory includes vector PDF and PNG plots; \texttt{data/} includes all consolidated CSVs; \texttt{provenance/} includes source commits, environment versions, source portability patches and audits; \texttt{logs/} retains the three failure logs. Clinical CSVs contain derived subject-level analysis outputs; the original public cohort files are not redistributed in this package. This package contains generated outputs and report sources, not a replacement for the complete training environments.

\begin{thebibliography}{9}
\bibitem{rankwalk} Pfeifer, B. \emph{Contrastive Representation Learning of Longitudinal Disease Trajectories on Temporal Graphs}. arXiv:2607.25609, uploaded reference PDF, 2026.
\bibitem{ts2vec} Yue, Z., et al. \emph{TS2Vec: Towards Universal Representation of Time Series}. AAAI, 2022. Official implementation: \url{https://github.com/yuezhihan/ts2vec}.
\bibitem{timesurl} Liu, J. and Chen, S. \emph{TimesURL: Self-Supervised Contrastive Learning for Universal Time Series Representation Learning}. AAAI 38(12), 13918--13926, 2024. \url{https://github.com/Alrash/TimesURL}.
\bibitem{vader} de Jong, J., et al. VaDER, \emph{GigaScience}, 2019, doi:10.1093/gigascience/giz134. \url{https://github.com/johanndejong/VaDER}.
\bibitem{pace} Wang, J.-L., Chiou, J.-M. and M\"uller, H.-G. Functional Data Analysis. \emph{Annual Review of Statistics and Its Application}, 2016. Native \texttt{fdapace} package; exact version in provenance.
\end{thebibliography}
\appendix
\clearpage\section{Example simulated trajectories}
''')
figure('S1_clean_trajectories.pdf','Fresh Simulation 1 replicate 1: all 200 subjects, colored by ground-truth cluster.')
figure('S2_regular_trajectories.pdf','Fresh Simulation 2 replicate 1: all 200 subjects, colored by ground-truth cluster.')
add(r'\clearpage\section{Source-input leakage diagnostics}')
rows=[]
for m in ['fPCA','DTW','kml3d','TAPIO_PC1','TAPIO_weighted','TS2Vec_repo']:
 z=sim[(sim.scenario=='S2_regular_repo_latent_DIAGNOSTIC')&(sim.method==m)].ARI;rows.append([label(m),str(len(z)),pm(z)])
table(['Method using source S2 inputs','Fits','ARI'],rows,'Source serialization includes unobserved simulator regime state. These are diagnostic reproduction results and are excluded from the observed-data competitor ranking.')
figure('S2_regular_repo_latent_DIAGNOSTIC.pdf','Leaked-input source-script diagnostic only. These values must not be interpreted as observed-data comparator performance.')
add(r'\end{document}')
(O/'report.tex').write_text('\n\n'.join(parts))
(O/'Makefile').write_text('all: report.pdf\n\nreport.pdf: report.tex $(wildcard figures/*.pdf)\n\tpdflatex -interaction=nonstopmode -halt-on-error report.tex\n\tpdflatex -interaction=nonstopmode -halt-on-error report.tex\n\nclean:\n\trm -f report.aux report.log report.out report.toc\n')
(O/'README.md').write_text('''# RankWalk final evaluation report

Open `report.pdf`. Edit `report.tex` and run `make` to rebuild. Requires a standard TeX Live installation with pdflatex, booktabs, geometry, graphicx, hyperref and Latin Modern. No shell escape or network access is required.

All file paths are relative. The figures directory includes every original evaluation plot plus final real-data extension plots, in PDF and PNG. Consolidated CSVs, provenance and the three VaDER failure logs are included. Scores are mean ± sample SD. Full-follow-up clinical concordance is descriptive association, not prospective accuracy. See the report for protocol differences and the failed-fit audit.

No changes were pushed to RankWalk. This is a frozen report snapshot of the completed compute queue, including documented numerical failures.
''')
print('LaTeX report created:',O/'report.tex')
