root <- '/workspace/rankwalk-paper-reproduction'
library(MASS)
source(file.path(root,'TAPIO/R/generate_random_curve.R'))
source(file.path(root,'TAPIO/R/simLongData.R'))
source('/workspace/RankWalk/experiments/simLongData_GNN.R')
original_sim <- simLongData
# Minimal external sensitivity: move random-curve draw outside visits and index it.
b <- paste(deparse(body(simLongData)),collapse='\n')
b <- sub('for (j in 1:n_i) {','random_curve <- generate_random_curve(times)\nfor (j in 1:n_i) {',b,fixed=TRUE)
b <- sub('random_effect_t <- generate_random_curve(t_ij)','random_effect_t <- random_curve[j]',b,fixed=TRUE)
subject_sim <- simLongData; body(subject_sim) <- parse(text=b)[[1]]
scenarios <- c('S1_clean','S1_noisy','S2_regular','S1_irregular','S2_irregular','S1_subject_curve')
for(s in scenarios) for(i in 1:50) {
  path <- file.path(root,'data',sprintf('%s_%02d.csv',s,i)); if(file.exists(path)) next
  set.seed(10000+i)
  sigma <- rep(if(s=='S1_noisy') 3 else 5,5)
  if(s=='S1_noisy') sigma[sample(1:5,1)]<-sample(3:20,1)
  d <- if(startsWith(s,'S2')) simLongData_hard(ranTimes=s=='S2_irregular') else if(s=='S1_subject_curve') subject_sim(ranTimes=FALSE,n_i=10,eta=3,sigma_diag=sigma) else original_sim(ranTimes=s=='S1_irregular',n_i=10,eta=3,sigma_diag=sigma)
  write.csv(d,path,row.names=FALSE)
  cat(s,i,'\n'); flush.console()
}
