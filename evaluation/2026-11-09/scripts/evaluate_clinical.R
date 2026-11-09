a<-commandArgs(TRUE);library(survival);library(Hmisc)
x<-read.csv(a[1],check.names=FALSE);surv<-read.csv(a[2]);seed<-as.integer(a[4]);k0<-as.integer(a[5]);ids<-x[[1]];features<-as.matrix(x[,-1,drop=FALSE]);features<-scale(features);features[!is.finite(features)]<-0
res<-list()
native<-length(a)>=6 && a[6]=="labels"
for(k in if(native) k0 else unique(c(k0,4))) {
 set.seed(seed);labels<-if(native) as.integer(x[[2]]) else kmeans(features,centers=k,nstart=100)$cluster
 d<-merge(surv,data.frame(id=ids,cluster=labels),by='id');d<-d[complete.cases(d),];risk<-tapply(d$time,d$cluster,mean);ord<-order(risk);mapping<-setNames(seq_along(ord),names(risk)[ord]);score<-as.numeric(mapping[as.character(d$cluster)])
 res[[as.character(k)]]<-data.frame(k=k,n=nrow(d),events=sum(d$event),C_index=as.numeric(rcorr.cens(score,Surv(d$time,d$event))['C Index']),logrank_chisq=if(length(unique(d$cluster))>1) survdiff(Surv(time,event)~factor(cluster),data=d)$chisq else NA_real_,observed_clusters=length(unique(d$cluster)),min_cluster=min(table(d$cluster)))
 write.csv(d,paste0(a[3],'_K',k,'_clusters.csv'),row.names=FALSE)
}
write.csv(do.call(rbind,res),paste0(a[3],'_metrics.csv'),row.names=FALSE)
