#' Cluster-specific features importance
#' 
#' @param res The object returned by TAPIO. 
#' @return Cluster-specific feature importance 
#'
#' @examples
#' NaN
#'
#'@export

importance <- function(res){
	
CL = res$cl

IMP <- array(0, c(length(CL), length(CL), dim(res$feature_importance)[2]));

FEATIMP = res$feature_importance

for(xx in 1:(length(CL)-1)){

cat(xx, " of " , length(CL)-1, " done ","\n")

	for(yy in (xx+1):length(CL)){

		if(CL[xx]==CL[yy]){

			for(zz in 1:length(res$PART)){

				no_nas = which(!is.na(FEATIMP[zz,]))

				val = rep(NaN,dim(FEATIMP)[2])
				#res$PART[[zz]] = 1 - res$PART[[zz]]/max(res$PART[[zz]]) 
				val[no_nas] = res$PART[[zz]][xx,yy]*FEATIMP[zz,][no_nas]				

				for(ii in 1:length(val)){
					
					if(!is.na(val[ii])){
						IMP[xx,yy,ii] = IMP[xx,yy,ii] + val[ii] 
						IMP[yy,xx,ii] = IMP[yy,xx,ii] + val[ii] 
					}
				}

			}

		}

	}
}

#return(IMP)

CLASSES = unique(CL)

CLASS_IMP = matrix(0,length(CLASSES),dim(res$feature_importance)[2])

for(xx in 1:length(CLASSES)){

    #print(CLASS_IMP)
	ids1 = which(res$cl==CLASSES[xx], arr.ind=TRUE)
	#ids2 = which(res$cl!=CLASSES[xx], arr.ind=TRUE)

		for(yy in 1:dim(res$feature_importance)[2]){

			#IMP2 = IMP
			#IMP2[IMP==0] = NaN
			#IMP[,,yy] = IMP[,,yy]/max(IMP[,,yy]) 
			
			CLASS_IMP[xx,yy] = CLASS_IMP[xx,yy] + mean(IMP[ids1,ids1,yy], na.rm=TRUE) 

		}

	CLASS_IMP[xx,] = CLASS_IMP[xx,]#/sum(CLASS_IMP[xx,])

}



return(CLASS_IMP/max(CLASS_IMP))

}