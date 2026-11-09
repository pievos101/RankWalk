root<-'/workspace/rankwalk-paper-reproduction';args<-commandArgs(TRUE)
if(length(args)&&args[1]=='new') {
 load('/workspace/rankwalk-research/data_sources/JM/data/prothro.rda');load('/workspace/rankwalk-research/data_sources/JM/data/prothros.rda')
 d<-data.frame(subject=as.numeric(prothro$id),time=prothro$time,outcome=1,y=log1p(prothro$pro));s<-data.frame(id=as.numeric(prothros$id),time=prothros$Time,event=prothros$death)
 write.csv(d,file.path(root,'data/clinical_PROTHRO_long.csv'),row.names=FALSE);write.csv(s,file.path(root,'data/clinical_PROTHRO_survival.csv'),row.names=FALSE)
 load(file.path(root,'data_sources/joineRML/data/renal.rda'));cols<-c('proteinuria','haematocrit','gfr');parts<-list()
 # Prespecified absolute-time aggregation controls the very dense renal cohort.
 end<-max(c(renal$prot$years,renal$haem$years,renal$gfr$years),na.rm=TRUE);edges<-seq(0,end+1e-8,length.out=33)
 for(j in 1:3){z<-renal[[j]];z$bin<-cut(z$years,edges,include.lowest=TRUE,labels=FALSE);z$y<-if(j==3) log1p(z[[cols[j]]]) else z[[cols[j]]];z<-z[is.finite(z$y),];a<-aggregate(cbind(y,years)~id+bin,z,mean);parts[[j]]<-data.frame(subject=a$id,time=(edges[a$bin]+edges[a$bin+1])/2,outcome=j,y=a$y)}
 d<-do.call(rbind,parts);s<-data.frame(id=renal$surv$id,time=renal$surv$fuyears,event=renal$surv$failure)
 write.csv(d,file.path(root,'data/clinical_RENAL_long.csv'),row.names=FALSE);write.csv(s,file.path(root,'data/clinical_RENAL_survival.csv'),row.names=FALSE)
} else {
 for(cohort in c('PBC2','AIDS','HEART','PAQUID')){
 p<-file.path(root,'results',paste0('clinical_',cohort,'_01.rds'));if(!file.exists(p)) next
 x<-readRDS(p);d<-x$long;if(cohort=='AIDS')d<-d[d$outcome==1,];s<-x$survival;if('ID'%in%names(s)) names(s)[names(s)=='ID']<-'id'
 if(cohort=='PAQUID'){
 load('/workspace/RankWalk/experiments/paquid.rda');raw<-paquid[paquid$ID%in%s$id,];s<-do.call(rbind,lapply(split(raw,raw$ID),function(z)data.frame(id=unique(z$ID),time=unique(z$agedem)[1]-min(z$age,na.rm=TRUE),event=unique(z$dem)[1])));s<-s[complete.cases(s)&s$time>0,];d<-d[d$subject%in%s$id,]
 }

 write.csv(d,file.path(root,'data',paste0('clinical_',cohort,'_long.csv')),row.names=FALSE);write.csv(s[,c('id','time','event')],file.path(root,'data',paste0('clinical_',cohort,'_survival.csv')),row.names=FALSE)
 }
}
