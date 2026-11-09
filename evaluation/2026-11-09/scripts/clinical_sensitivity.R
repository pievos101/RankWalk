root<-'/workspace/rankwalk-paper-reproduction';library(survival);library(Hmisc)
for(cohort in c('PBC2','AIDS','HEART','PAQUID')) for(i in 1:if(cohort=='PAQUID')30 else 20){
 file<-file.path(root,'results',sprintf('clinical_%s_%02d.rds',cohort,i));out<-file.path(root,'results',sprintf('sensitivity_%s_%02d.csv',cohort,i));if(!file.exists(file)||file.exists(out))next
 x<-readRDS(file);s<-x$survival;if('ID'%in%names(s))names(s)[names(s)=='ID']<-'id';res<-list()
 for(method in c('FPCA','RankWalk'))for(k in unique(c(if(cohort=='HEART')2 else 3,4))){features<-x[[method]];features<-scale(features);features[!is.finite(features)]<-0;set.seed(20000+i);cl<-kmeans(features,k,nstart=100)$cluster;ids<-as.numeric(rownames(features));z<-merge(s,data.frame(id=ids,cluster=cl),by='id');risk<-tapply(z$time,z$cluster,mean);mapping<-setNames(seq_along(risk),order(risk));score<-as.numeric(mapping[as.character(z$cluster)]);res[[paste(method,k)]]<-data.frame(cohort=cohort,replicate=i,method=method,k=k,n=nrow(z),C_index=as.numeric(rcorr.cens(score,Surv(z$time,z$event))['C Index']),logrank_chisq=survdiff(Surv(time,event)~factor(cluster),data=z)$chisq)}
 write.csv(do.call(rbind,res),out,row.names=FALSE)
}
source('/workspace/rankwalk-paper-reproduction/scripts/paquid_agedem.R')
