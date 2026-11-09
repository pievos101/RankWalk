# Reproduce source Simulation 2 feature leakage as a labeled diagnostic only.
root<-'/workspace/rankwalk-paper-reproduction';Sys.setenv(RGL_USE_NULL='TRUE');library(TAPIO);library(fda);library(cluster);library(dtw);library(kml3d)
ari<-function(x,y){tab<-table(x,y);ch<-function(z)z*(z-1)/2;a<-sum(ch(tab));b<-sum(ch(rowSums(tab)));c<-sum(ch(colSums(tab)));e<-b*c/ch(sum(tab));(a-e)/((b+c)/2-e)}
for(i in 1:50){file<-file.path(root,'data',sprintf('S2_regular_%02d.csv',i));while(!file.exists(file))Sys.sleep(5);d<-read.csv(file);wide<-reshape(d,idvar=c('subject','time','cluster'),timevar='outcome',direction='wide');x<-as.matrix(wide[,4:ncol(wide)]);truth<-aggregate(wide$cluster,list(wide$subject),function(x)x[1])[,2];traj<-array(NA,c(200,10,5));for(m in 1:5) for(t in 1:10)traj[,t,m]<-x[seq(t,nrow(x),10),m]
 for(method in c('fPCA','DTW','TAPIO_PC1','TAPIO_weighted','kml3d')){
 out<-file.path(root,'results',sprintf('S2_regular_%02d_%s_repo_latent.csv',i,method));if(file.exists(out))next
 set.seed(10000+i);start<-proc.time()[3]
 tryCatch({pred<-switch(method,
 fPCA={basis<-create.bspline.basis(c(1,10),nbasis=5);scores<-do.call(cbind,lapply(1:5,function(m)pca.fd(smooth.basis(1:10,t(traj[,,m]),basis)$fd,nharm=2)$scores));kmeans(scores,4,nstart=25)$cluster},
 DTW={seqs<-lapply(unique(wide$subject),function(s)x[wide$subject==s,,drop=FALSE]);distances<-proxy::dist(seqs,method=function(a,b)dtw::dtw(a,b,distance.only=TRUE)$distance);pam(as.dist(distances),4)$clustering},
 TAPIO_PC1=longTAPIO_trajectories(x,k=4,user_id=wide$subject,levels=4,n_trees=500,method='ward.D2',n_features=NaN,do.leveling=TRUE,pca_selection='first')$cl,
 TAPIO_weighted=longTAPIO_trajectories(x,k=4,user_id=wide$subject,levels=4,n_trees=500,method='ward.D2',n_features=NaN,do.leveling=TRUE,pca_selection='random_weighted')$cl,
 kml3d={obj<-clusterLongData3d(traj=traj,idAll=as.character(1:200),time=1:10,varNames=paste0('Marker',1:5));kml3d(obj,nbClusters=4,nbRedrawing=10,toPlot='none',parAlgo=parKml3d(imputationMethod='copyMean'));getClusters(obj,4)})
 write.csv(data.frame(scenario='S2_regular_repo_latent_DIAGNOSTIC',replicate=i,method=method,seed=10000+i,ARI=ari(truth,pred),seconds=proc.time()[3]-start),out,row.names=FALSE)
 },error=function(e)cat('FAIL',i,method,conditionMessage(e),'\n'))
 }
}
