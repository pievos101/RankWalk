import os
os.environ['MPLCONFIGDIR']='/workspace/rankwalk-paper-reproduction/cache/matplotlib'
import json,datetime,hashlib,subprocess,html,sys
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import fcntl
R=Path('/workspace/rankwalk-paper-reproduction');lock_handle=(R/'report.lock').open('w');fcntl.flock(lock_handle,fcntl.LOCK_EX)
O=R/'report';O.mkdir(exist_ok=True);F=O/'figures';F.mkdir(exist_ok=True)
rows=[];ext=[]
for p in (R/'results').glob('*.json'):
 try:d=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 if 'ARI' in d:rows.append(d)
 if p.name.startswith('extension_'):ext.append(d)
for p in (R/'results').glob('S*.csv'):
 if any(z in p.name for z in ['features','clusters']):continue
 try:d=pd.read_csv(p)
 except Exception:continue
 if 'ARI' in d:rows.extend(d.to_dict('records'))
df=pd.DataFrame(rows);df.to_csv(O/'simulation_results.csv',index=False)
styles=getSampleStyleSheet();styles['BodyText'].fontSize=9;styles['BodyText'].leading=12
story=[];web=[]
def p(text,style='BodyText'):
 story.append(Paragraph(text,styles[style]));story.append(Spacer(1,7));web.append('<p>'+text+'</p>')
def table(data):
 t=Table(data,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efed')),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.5,colors.grey)]));story.append(t);story.append(Spacer(1,9));web.append(pd.DataFrame(data[1:],columns=data[0]).to_html(index=False))
def fig(name):
 from PIL import Image as PILImage
 with PILImage.open(F/(name+'.png')) as picture: aspect=picture.height/picture.width
 story.append(Image(str(F/(name+'.png')),width=470,height=470*aspect));story.append(Spacer(1,7));web.append(f'<img src="figures/{name}.png">')
subprocess.run([sys.executable,str(R/'scripts/validate_progress.py')],stdout=subprocess.DEVNULL,check=True)
status=json.loads((R/'progress_validation.json').read_text()) if (R/'progress_validation.json').exists() else {}
complete=status.get('complete',False)
finished=(R/'queue_finished.txt').exists()
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
p('RankWalk: paper reproduction and longitudinal benchmark extension','Title');p(('COMPLETED REPORT — ' if complete else 'FINAL REPORT — WITH REPORTED METHOD FAILURES — ' if finished else 'LIVE INTERMEDIATE REPORT — ')+now,'Heading2')
p('Repository settings are the primary specification, with paper-specification sensitivities reported separately. Fresh results only. The compute queue has finished. All 300 RankWalk simulations and all 90 original clinical fits completed. Three VaDER fits failed despite retries: Simulation 2 replicate 9, RENAL replicate 8, and HEART replicate 7. Their numerical failures are retained in failed_runs.json and logs; summaries use successful fits only, with explicit counts. No original paper scores are substituted for new runs.')

for cohort in ['PROTHRO','RENAL','PBC2','AIDS','HEART','PAQUID']:
 t=[d for d in ext if d['cohort']==cohort and d['method']=='TimeOnly'];g=[d for d in ext if d['cohort']==cohort and d['method']=='RankWalk']
 if not t or not g:continue
 k=2 if cohort=='HEART' else 3
 tv=np.mean([next(m['C_index'] for m in d['metrics'] if m['k']==k) for d in t]);gv=np.mean([next(m['C_index'] for m in d['metrics'] if m['k']==k) for d in g])
 if tv>gv:
  p(f'Observation-process control: {cohort}, K={k}, interim TimeOnly C-index {tv:.3f} ({len(t)} fits), RankWalk {gv:.3f} ({len(g)} fits). Strong full-follow-up concordance is achievable without biomarker values. This limits prognostic interpretation of these association metrics. A fixed-landmark, held-out analysis would be needed to establish predictive accuracy.')
p('Provenance and interpretation','Heading2')
p('RankWalk source: commit 58a87af4fdcdd243425106622e0e4f46dcc1ba6a; CPU Numba backend, sparse Jaccard, fixed unique-node walk sampling. All work is outside the repository. The uploaded 12-page paper is the reference; arXiv could not be fetched through the environment network policy. Statistical packages are native R implementations, not Python approximations.')
p('Simulation target: N=200, four balanced clusters, five outcomes; 50 replicates per scenario. Seeds 10001–10050 are shared between paired methods. Native clinical scripts seed R clustering, Torch initialization and walk sampling, but their NumPy graph subspace RNG was not pre-seeded; saved representations support exact metric replay, while graph refits are Monte Carlo reruns. Each method gets a fresh method-specific seed reset. The published original seeds are unavailable, so exact numerical equality is not expected. RankWalk: 100 epochs, walk 20, top-k 10, kNN 5, R KMeans 50 starts. TS2Vec: official implementation, 50 epochs and 64 output dimensions. VaDER: official TensorFlow implementation, [12,2] hidden/latent dimensions, 50 pretraining + 100 fitting epochs, native mixture labels. Native fPCA, TAPIO, kml3d and DTW retain the repository budgets.')
p('Clinical metrics use complete observed follow-up and post-hoc cluster risk ordering. They measure descriptive cluster–outcome association, not independent predictive accuracy or AUROC. Repeated fits quantify algorithmic variability, not patient-sampling uncertainty. Runtime values were collected during concurrent execution and are not isolated speed comparisons.')
p('Specification differences and explicit corrections','Heading2')
for text in [
'Simulation 1: primary code redraws quadratic random-curve coefficients at each visit. A sensitivity draws a single random curve per subject, as described in the paper. Clean sigma=5; corrupted-marker sigma starts at 3, with one marker sampled uniformly from integer 3…20. These match the figure settings rather than the currently active sigma=7 in the script.',
'Irregular simulations: the source flattens variable-length embeddings into rows of unequal lengths. Reproduction uses the same irregular graph settings, followed by interpolation of node embeddings onto a common ten-point absolute-time grid before flattening. This is an explicit compatibility correction; results are not an exact execution of the invalid ragged-row operation.',
'TS2Vec: the repository serializes long rows as 50 positions × 2 columns (outcome identifier and value). This is retained as TS2Vec_repo for Simulation 1, with a proper 10 × 5 observed-marker sensitivity. Simulation 2 adds an unobserved latent regime column to that wrapper and the native R wide matrix. Consequently the source TAPIO/DTW use latent state, and the source fPCA/kml3d select a mixture of outcomes and latent state in their first five columns. Observed-marker versions are the valid comparisons. Native R source-input versions are shown separately as leaked-input diagnostics, never as valid competitor evidence. The latent-state TS2Vec version is excluded.',
'Clinical primary source settings: PBC/AIDS/PAQUID K=3 and HEART K=2. K=4 is a paper-spec sensitivity. PBC/HEART/AIDS use mean node pooling, while PAQUID uses zero-padded flattened embeddings; primary scripts preserve those choices. Actual source calls use 100 GNN epochs for PBC/HEART/PAQUID and 200 for AIDS. AIDS also duplicates the CD4 channel in the source; primary reproduction retains this duplication. Undefined standardized constant padding columns are replaced with zero.',
'PAQUID primary code derives dementia time from the first observed row marked dem=1 and uses fitted categorical Cox concordance. Because dem is a subject-level eventual-status indicator, this assigns time zero to dementia cases. That source endpoint is clinically invalid as incident-dementia follow-up, even if its numerical scores reproduce. This also differs from risk-ordered Hmisc concordance in the other cohorts. Agedem-based onset and common metric sensitivities must be distinguished from this native result.',
'New cohorts: PROTHRO (cirrhosis trial) and RENAL (renal transplantation). RENAL has more than 180,000 marker records; absolute follow-up is aggregated into 32 prespecified bins before graph/FPCA analysis. Comparator encoders use common 32-point absolute-time interpolation and global marker standardization. TimeOnly clusters visit-time mean, range, standard deviation and visit count; ValueSummary clusters marker means, SDs, first/last values and slopes. These controls assess whether follow-up structure or simple trajectory summaries explain the association. All extension methods use the same eligible observed subjects, without outcome labels during representation fitting. The PAQUID extension uses the agedem endpoint correction rather than the source time-zero endpoint; AIDS extension uses a single CD4 channel rather than the source duplication.']:
 p(text)

p('Example simulated trajectories','Heading2')
for scenario,title in [('S1_clean','Simulation 1'),('S2_regular','Simulation 2')]:
 name=scenario+'_trajectories';path=R/'data'/(scenario+'_01.csv')
 if not path.exists():continue
 if not (F/(name+'.png')).exists():
  z=pd.read_csv(path);figobj,axes=plt.subplots(2,3,figsize=(12,6));palette=plt.get_cmap('Set2')
  for ax,marker in zip(axes.flat,sorted(z.outcome.unique())):
   for sid,sub in z[z.outcome==marker].groupby('subject'):
    sub=sub.sort_values('time');ax.plot(sub.time,sub.y,color=palette(int(sub.cluster.iloc[0])-1),alpha=.14,lw=.6)
   ax.set_title('Outcome '+str(marker));ax.set_xlabel('Time');ax.set_ylabel('Observed value');ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.2)
  axes.flat[-1].set_visible(False);figobj.suptitle(title+' — fresh replicate 1, all 200 subjects');figobj.tight_layout();figobj.savefig(F/(name+'.png'),dpi=150,bbox_inches='tight');figobj.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(figobj)
 fig(name)
p('Fresh simulation results','Heading2')
labels={'S1_clean':'Simulation 1, clean (sigma 5)','S1_noisy':'Simulation 1, one corrupted marker','S2_regular':'Simulation 2, regime switching','S1_irregular':'Simulation 1, irregular','S2_irregular':'Simulation 2, irregular','S1_subject_curve':'Simulation 1, subject-shared random curve','S2_regular_repo_latent_DIAGNOSTIC':'Simulation 2 source-input diagnostic: includes latent regime'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white'})
if len(df):
 summary=df.groupby(['scenario','method']).ARI.agg(['count','mean','std','median']).reset_index();summary.to_csv(O/'simulation_summary.csv',index=False)
 for scenario,title in labels.items():
  d=df[df.scenario==scenario]
  if d.empty:continue
  p(title,'Heading3');summ=summary[summary.scenario==scenario].sort_values('mean',ascending=False)
  table([['Method','Completed / target','Mean ARI','SD']]+[[r.method,f'{r.count} / 50',f'{r.mean:.3f}',f'{r.std:.3f}' if pd.notna(r.std) else '—'] for r in summ.itertuples(index=False)])
  order=d.groupby('method').ARI.median().sort_values().index.tolist();figobj,axs=plt.subplots(1,2,figsize=(12,6));palette=plt.get_cmap('Set2');rng=np.random.default_rng(0)
  for j,m in enumerate(order):
   v=d[d.method==m].ARI.to_numpy();color=palette(j%8)
   if len(v)>1 and np.ptp(v)>1e-8:
    body=axs[0].violinplot([v],positions=[j],orientation="horizontal",showextrema=False)['bodies'][0];body.set_facecolor(color);body.set_alpha(.5)
   axs[0].boxplot([v],positions=[j],orientation="horizontal",widths=.18,showfliers=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.7})
   axs[0].scatter(v,j+rng.uniform(-.12,.12,len(v)),color='black',s=8,alpha=.4)
  axs[0].set_yticks(range(len(order)),order);axs[0].set_xlabel('Adjusted Rand index');axs[0].set_title('Fresh replicates; target n=50 per method')
  pivot=d.pivot(index='replicate',columns='method',values='ARI');dom=np.full((len(order),len(order)),np.nan);counts=np.zeros_like(dom)
  for yi,a in enumerate(order):
   for xi,b in enumerate(order):
    if a==b:continue
    z=pivot[[a,b]].dropna();ties=np.isclose(z[a],z[b],atol=1e-12,rtol=0);z=z[~ties];counts[yi,xi]=len(z)
    if len(z):dom[yi,xi]=(z[b]>z[a]).mean()
  cmap=LinearSegmentedColormap.from_list('dominance',['#b2182b','#ffffff','#006837']);cmap.set_bad('#d8d8d8');im=axs[1].imshow(dom,vmin=0,vmax=1,cmap=cmap)
  axs[1].set_xticks(range(len(order)),order,rotation=45,ha='right');axs[1].set_yticks(range(len(order)),order);axs[1].set_xlabel('Column method beats row method');axs[1].set_title('Paired dominance; ties excluded');axs[1].grid(False)
  for yi in range(len(order)):
   for xi in range(len(order)):
    if np.isfinite(dom[yi,xi]):axs[1].text(xi,yi,f'{dom[yi,xi]:.2f}\n(n={int(counts[yi,xi])})',ha='center',va='center',fontsize=7)
  figobj.suptitle(title);figobj.tight_layout();name=scenario;figobj.savefig(F/(name+'.png'),dpi=160,bbox_inches='tight');figobj.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(figobj);fig(name)
  p('Dominance is computed only on paired completed replicates, separately for each pair. Small interim counts can produce apparent perfect dominance and should not be treated as established superiority.')

p('Regular versus irregular RankWalk','Heading2')
if len(df):
 for a,b,title in [('S1_clean','S1_irregular','Simulation 1'),('S2_regular','S2_irregular','Simulation 2')]:
  z=df[(df.method=='RankWalk') & df.scenario.isin([a,b])]
  if z.scenario.nunique()<2:continue
  figobj,ax=plt.subplots(figsize=(10,4));vals=[z[z.scenario==q].ARI for q in [a,b]]
  bp=ax.boxplot(vals,orientation='horizontal',tick_labels=['Regular','Irregular'],patch_artist=True,showfliers=False)
  for patch,c in zip(bp['boxes'],['#66c2a5','#fc8d62']):patch.set_facecolor(c)
  ax.set_xlabel('Adjusted Rand index');ax.set_title(title+' — fixed fast RankWalk; target 50 fits each');figobj.tight_layout();name=a+'_vs_irregular';figobj.savefig(F/(name+'.png'),dpi=160,bbox_inches='tight');figobj.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(figobj);fig(name)
p('Original-cohort clinical reproduction','Heading2')
original={'PBC2':(.78,.02,246.91,20.85,.76,127.50),'AIDS':(.69,0,105.86,6.80,.64,53.09),'HEART':(.71,.02,52.25,7.96,.57,4.81),'PAQUID':(.70,.02,39.80,6.64,.66,5.06)}
clinicalrows=[]
for cohort,published in original.items():
 path=R/'results'/f'clinical_{cohort}.csv'
 if not path.exists():continue
 x=pd.read_csv(path).dropna();target=30 if cohort=='PAQUID' else 20
 if not len(x):continue
 x.columns=['FPCA_C','FPCA_LR','GNN_C','GNN_LR']
 p(f'{cohort}: {len(x)} / {target} completed','Heading3')
 table([['Method','Fresh C-index','Fresh log-rank chi²','Published C / chi²'],['RankWalk',f'{x.GNN_C.mean():.3f} ± {x.GNN_C.std():.3f}',f'{x.GNN_LR.mean():.2f} ± {x.GNN_LR.std():.2f}',f'{published[0]:.2f} / {published[2]:.2f}'],['FPCA',f'{x.FPCA_C.mean():.3f}',f'{x.FPCA_LR.mean():.2f}',f'{published[4]:.2f} / {published[5]:.2f}']])
 for method,cc,lr in [('RankWalk','GNN_C','GNN_LR'),('FPCA','FPCA_C','FPCA_LR')]:
  for rep,row in x.iterrows():clinicalrows.append(dict(cohort=cohort,method=method,replicate=rep+1,C_index=row[cc],logrank_chisq=row[lr],scope='native repository metric'))
 figobj,axs=plt.subplots(1,2,figsize=(12,4))
 for ax,col,label in [(axs[0],'C','C-index'),(axs[1],'LR','Log-rank chi-square')]:
  vals=[x['FPCA_'+col],x['GNN_'+col]];bp=ax.boxplot(vals,tick_labels=['FPCA','RankWalk'],patch_artist=True,showfliers=False)
  for patch,c in zip(bp['boxes'],['#66c2a5','#fc8d62']):patch.set_facecolor(c)
  for j,v in enumerate(vals):ax.scatter(np.full(len(v),j+1)+np.random.default_rng(j).uniform(-.08,.08,len(v)),v,color='black',alpha=.3,s=9)
  ax.set_ylabel(label)
 figobj.suptitle(cohort+' — fresh native reproduction');figobj.tight_layout();figobj.savefig(F/(cohort+'.png'),dpi=160,bbox_inches='tight');figobj.savefig(F/(cohort+'.pdf'),bbox_inches='tight');plt.close(figobj);fig(cohort)
pd.DataFrame(clinicalrows).to_csv(O/'clinical_primary_results.csv',index=False)
p('Real-world extensions','Heading2');extensionrows=[]
for d in ext:
 for m in d['metrics']:extensionrows.append(dict(cohort=d['cohort'],method=d['method'],replicate=d['replicate'],**m))
if extensionrows:
 e=pd.DataFrame(extensionrows);e.to_csv(O/'clinical_extension_results.csv',index=False)
 for cohort in e.cohort.unique():
  z=e[e.cohort==cohort];p(cohort,'Heading3');g=z.groupby(['method','k']).agg(n_fits=('C_index','size'),C_index=('C_index','mean'),chisq=('logrank_chisq','mean'),subjects=('n','min')).reset_index()
  table([['Method','K','Fits / 10','Subjects','Mean C-index','Mean chi²']]+[[r.method,str(r.k),f'{r.n_fits} / 10',str(r.subjects),f'{r.C_index:.3f}',f'{r.chisq:.2f}'] for r in g.itertuples(index=False)])
else:p('No extension method has completed yet; no results are inferred.')
p('Clinical sensitivity checks','Heading2')
for pattern,title in [('sensitivity_*.csv','Source K versus paper K=4, common Hmisc concordance'),('agedem_*.csv','PAQUID onset-age endpoint correction')]:
 parts=[pd.read_csv(z) for z in (R/'results').glob(pattern)]
 if not parts:continue
 z=pd.concat(parts,ignore_index=True);z.to_csv(O/(pattern.replace('*','all')),index=False);p(title,'Heading3')
 g=z.groupby(['cohort','method','k']).agg(fits=('C_index','size'),C_index=('C_index','mean'),chisq=('logrank_chisq','mean')).reset_index()
 table([['Cohort','Method','K','Fits','Mean C-index','Mean chi²']]+[[r.cohort,r.method,str(r.k),str(r.fits),f'{r.C_index:.3f}',f'{r.chisq:.2f}'] for r in g.itertuples(index=False)])
p('Methods and sources','Heading2');p('Pfeifer, Contrastive Representation Learning of Longitudinal Disease Trajectories on Temporal Graphs, arXiv:2607.25609 (uploaded PDF). TS2Vec: Yue et al., AAAI 2022, github.com/yuezhihan/ts2vec. TimesURL: Liu and Chen, AAAI 2024, github.com/alrash/TimesURL; the two CPU tensor-allocation portability edits are recorded in the source patch. VaDER: de Jong et al., GigaScience 2019, github.com/johanndejong/VaDER. These are selected established competitors, not an exhaustive evaluation of every 2026 method. Public cohort sources and pinned commits are listed in manifest.json.');p('Completion status and reproducibility','Heading2');p('All expected runs and sensitivity counts passed the completeness checks.' if complete else 'The queue has finished with three documented VaDER numerical failures; the strict all-successful completeness flag remains false. All other planned run counts are complete.' if finished else 'Full reproduction is still running. Completed counts are shown explicitly; interim rankings are provisional.');p('The live CSV files contain every completed run, including seed and method. All scripts checkpoint and can resume without replacing existing results. Full completion requires all 50 simulation replicates, original-cohort 20/30 fits, and ten paired stochastic extension fits per method/cohort. Paper-style PNG and vector PDF figures are in figures/. Paper-spec K=4 and common clinical metric results are distinct from native primary metrics. Representation hyperparameters are fixed in advance; no outcome-based model selection is performed. Failed runs must remain visible in logs and must not be silently dropped.')
SimpleDocTemplate(str(O/'RankWalk_Paper_Reproduction.building.pdf'),title='RankWalk: Paper Reproduction and Longitudinal Benchmark Extension',rightMargin=40,leftMargin=40,topMargin=38,bottomMargin=38).build(story)
(O/'RankWalk_Paper_Reproduction.building.pdf').replace(O/'RankWalk_Paper_Reproduction.pdf')
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RankWalk reproduction</title><style>body{max-width:1100px;margin:40px auto;font:16px system-ui;line-height:1.55;color:#193330}img{max-width:100%}table{border-collapse:collapse}td,th{padding:7px 12px;border-bottom:1px solid #ddd}th{background:#e5efed}p{margin:12px 0}</style>'+''.join(web))
print(json.dumps(dict(generated=now,simulation_evaluations=len(df),clinical_fits=len(clinicalrows)//2,extension_fits=len(ext),report=str(O/'RankWalk_Paper_Reproduction.pdf'))))
