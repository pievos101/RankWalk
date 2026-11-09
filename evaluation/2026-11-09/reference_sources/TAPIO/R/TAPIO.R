#' Hierarchical clustering with an ensemble of PCA trees
#'
#' 
#' @param DATA The input data (rows: samples, columns: features) 
#' @param k Number of clusters.
#' @param n_features Number of features to sample (default: sqrt(ncolums))
#' @param n_trees Number of trees to grow
#' @param do.pca Dimension reduction using PCA (default=TRUE)
#' @param do.MFA Multi-View clustering (default=FALSE)
#' @param do.leveling Leveling (default=TRUE)
#' @param levels Number of levels to cut the dendrogram
#' @param max.k Number of maximum clusters when k=NaN
#' @param pca_selection "first" or "random_weighted"
#' @return The cluster solution 
#'
#' @examples
#' NaN
#'
#'@export


TAPIO <- function(DATA, k=NaN, n_features=NaN, n_trees=500, 
						do.pca=TRUE, do.MFA=FALSE, do.leveling=TRUE, 
						levels=10, max.k=10, pca_selection="first"){

	if(ncol(DATA)==2){
		#n_features = 2
	}

	if(is.list(DATA)){
		do.pca = FALSE
		do.MFA = TRUE
	}

	if(do.MFA){
		group = sapply(DATA, ncol)
		group2 = sort(rep(1:length(DATA), group))
		group3 = tapply(1:sum(group),group2,list)
		DATA = Reduce('cbind', DATA)
		DATA = as.data.frame(DATA)
	}

	if(is.na(n_features)){

		n_features = floor(sqrt(ncol(DATA)))

	}

	PART = vector("list", n_trees)
	IMP  = vector("list", n_trees)

	for (xx in 1:n_trees){

		ids    = sample(1:ncol(DATA), n_features, replace=TRUE)
		#ids_no = (1:ncol(DATA))[-ids]

		if(do.MFA){
			new_ids = list()
			for(zz in 1:length(group3)){
				new_ids[[zz]] = sample(group3[[zz]], n_features, replace=TRUE)
			}
		ids = unlist(new_ids)
		}

		DATA_s = DATA[,ids]
		#print(DATA_s)
		# PCA
		if(do.pca){
			# In case of missing values - us missMDA package
			if(sum(is.na(DATA_s))){
				warning("Use of missMDA package for Imputation")
				require(missMDA)
				res.comp = imputePCA(DATA_s, ncp=1)
				res.pca = prcomp(res.comp$completeObs, scale=FALSE)
			}else{
				res.pca = prcomp(DATA_s, scale=FALSE)
			}	
			var.cor = t(apply(res.pca$rotation, 1, var_cor_func, res.pca$sdev))
			var = .get_pca_var_results(var.cor)

			if(pca_selection == "first"){

 			   sel = 1

			}
			
			if(pca_selection == "random_weighted"){

    			eig_vals = res.pca$sdev^2
    			prob = eig_vals / sum(eig_vals)

    			sel = sample(1:length(prob), size = 1, prob = prob)

			}

			IMP[[xx]] = var$contrib[,sel]
			#IMP[[xx]][ids_no] = NaN 
			DATA_s = res.pca$x[,sel] # first PCA
		}

		if(do.MFA){

			# See whether there is multi-modal data
			mm = group2[ids]
			tt = table(mm)
			res.MFA = FactoMineR::MFA(DATA_s, tt, graph=FALSE)
			mfa.h = res.MFA$global.pca$ind$coord
			mfa.w = res.MFA$quanti.var$coord
			DATA_s = mfa.h[,1]
			IMP[[xx]] = res.MFA$global.pca$var$contrib[,1]
		}

		# LEVELING
		if(do.leveling){
			LEVELS = vector("list", levels)
			hc = fastcluster::hclust(dist(DATA_s), method="ward.D2")
			for(yy in 1:length(LEVELS)){
				cl = cutree(hc, yy+1)
				LEVELS[[yy]] = association(cl)
			}
			PART[[xx]] = Reduce("+",LEVELS)#/length(levels)
		
		# NO LEVELING
		}else{
			#hc = fastcluster::hclust(dist(DATA_s), method="ward.D2")
			#cl = cutree(hc, 2)
			#PART[[xx]] = HCfused::association(cl)
			PART[[xx]] = 1-dist(DATA_s)
		}
	}

	AFF  = Reduce("+", PART)
	DIST = 1 - AFF/max(AFF)

	# Final clustering
	hc = fastcluster::hclust(as.dist(DIST), method="ward.D2")
	
	if(is.na(k)){
		# find best k with Silhouette
		print("TAPIO::Silhouette")
		sil   <- calc.SIL(as.dist(DIST), size=max.k, method="ward.D2")
		#print(sil)
		id    <- which.max(sil)
		k     <- as.numeric(names(sil)[id])
		cl = cutree(hc, k)
	}else{
		cl = cutree(hc, k)
	}

	# Get the Importances
	Importance = matrix(NaN, n_trees, ncol(DATA))

	for (xx in 1:n_trees){

		imp = IMP[[xx]]
		ids = match(names(imp), colnames(DATA))
		#ids = match(names(imp), NN)
		Importance[xx,ids] = imp

	}

	#Importance = colMeans(Importance)

	return(list(cl=cl, PART=PART, feature_importance=Importance, DIST=DIST))

}


# compute all the results for variables : coord, cor, cos2, contrib
# var.coord : coordinates of variables on the principal component
.get_pca_var_results <- function(var.coord){
  
  var.cor <- var.coord # correlation
  var.cos2 <- var.cor^2 # variable qualities 
  
  # variable contributions (in percent)
  # var.cos2*100/total Cos2 of the component
  comp.cos2 <- apply(var.cos2, 2, sum)
  contrib <- function(var.cos2, comp.cos2){var.cos2*100/comp.cos2}
  var.contrib <- t(apply(var.cos2,1, contrib, comp.cos2))
  
  colnames(var.coord) <- colnames(var.cor) <- colnames(var.cos2) <-
    colnames(var.contrib) <- paste0("Dim.", 1:ncol(var.coord)) 
  
  # Variable coord, cor, cos2 and contrib
  list(coord = var.coord, cor = var.cor, cos2 = var.cos2, contrib = var.contrib)
}


# Correlation of variables with the principal component
var_cor_func <- function(var.loadings, comp.sdev){var.loadings*comp.sdev}
