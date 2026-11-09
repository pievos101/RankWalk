root <- '/workspace/rankwalk-paper-reproduction';Sys.setenv(RGL_USE_NULL='TRUE');library(MASS);library(fda);library(cluster);library(dtw);library(TAPIO);library(kml3d)
ari <- function(x,y) { tab<-table(x,y);n<-sum(tab);choose2<-function(z) z*(z-1)/2;a<-sum(choose2(tab));b<-sum(choose2(rowSums(tab)));c<-sum(choose2(colSums(tab)));expected<-b*c/choose2(n);(a-expected)/((b+c)/2-expected) }
scenarios<-c('S1_clean','S1_noisy','S2_regular','S1_subject_curve')
for(i in 1:50) for(s in scenarios){
 file<-file.path(root,'data',sprintf('%s_%02d.csv',s,i));while(!file.exists(file)) Sys.sleep(2)
 d<-read.csv(file);# Explicit observed outcomes; discard simulator-only latent regime.
 d<-d[,c('subject','time','outcome','y','cluster')]
 wide<-reshape(d,idvar=c('subject','time','cluster'),timevar='outcome',direction='wide');wide<-wide[order(wide$subject,wide$time),]
 x<-as.matrix(wide[,paste0('y.',1:5)]);ids<-sort(unique(wide$subject));truth<-sapply(ids,function(id) unique(wide$cluster[wide$subject==id]));n<-length(ids)
 traj<-array(NA,c(n,10,5));for(h in 1:5) for(t in 1:10) traj[,t,h]<-x[seq(t,nrow(x),10),h]
 methods<-if(s=='S1_subject_curve') 'fPCA' else c('fPCA','DTW','TAPIO_PC1','TAPIO_weighted','kml3d')
 for(method in methods){
 out<-file.path(root,'results',sprintf('%s_%02d_%s.csv',s,i,method));if(file.exists(out)) next
 set.seed(10000+i);start<-proc.time()[3]
 tryCatch({
 pred<-switch(method,
 fPCA={basis<-create.bspline.basis(c(1,10),nbasis=5);scores<-do.call(cbind,lapply(1:5,function(m) pca.fd(smooth.basis(1:10,t(traj[,,m]),basis)$fd,nharm=2)$scores));kmeans(scores,4,nstart=25)$cluster},
 DTW={seqs<-lapply(ids,function(id) x[wide$subject==id,,drop=FALSE]);distances<-proxy::dist(seqs,method=function(a,b) dtw::dtw(a,b,distance.only=TRUE)$distance);cluster::pam(as.dist(distances),k=4)$clustering},
 TAPIO_PC1=longTAPIO_trajectories(x,k=4,user_id=wide$subject,levels=4,verbose=0,n_trees=500,method='ward.D2',n_features=NaN,do.leveling=TRUE,pca_selection='first')$cl,
 TAPIO_weighted=longTAPIO_trajectories(x,k=4,user_id=wide$subject,levels=4,verbose=0,n_trees=500,method='ward.D2',n_features=NaN,do.leveling=TRUE,pca_selection='random_weighted')$cl,
 kml3d={obj<-clusterLongData3d(traj=traj,idAll=as.character(ids),time=1:10,varNames=paste0('Marker',1:5));kml3d(obj,nbClusters=4,nbRedrawing=10,toPlot='none',parAlgo=parKml3d(imputationMethod='copyMean'));getClusters(obj,4)})
 result<-data.frame(scenario=s,replicate=i,method=method,seed=10000+i,ARI=ari(truth,pred),seconds=proc.time()[3]-start)
 write.csv(result,out,row.names=FALSE);print(result);flush.console()
 },error=function(e){cat('FAIL',s,i,method,conditionMessage(e),'\n');flush.console()})
 }
}
