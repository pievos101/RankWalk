# =========================================================
# LIBRARIES
# =========================================================

library(JMbayes2)
library(survival)
library(Hmisc)
library(ggplot2)
library(fdapace)
library(reticulate)
library(gridExtra)



# =========================================================
# PYTHON RANKWALK WRAPPER
# =========================================================

Sys.setenv(
  RETICULATE_PYTHON = path.expand("~/rankwalk-venv/bin/python")
)

use_python(
  Sys.getenv("RETICULATE_PYTHON"),
  required = TRUE
)



py_run_string("

import numpy as np
import pandas as pd
import torch

from rankwalk import (
    build_temporal_graph_grid,
    compute_jaccard_fast,
    train_gnn
)



def run_rankwalk_gnn(df, epochs=120):


    df = pd.DataFrame(df)



    G, _ = build_temporal_graph_grid(

        df,

        n_bins=5,

        k_similarity=5,

        n_subspaces=5,

        subspace_ratio=0.7,

        overlap=0.5

    )



    nodes = list(G.nodes())



    device = torch.device(

        'cuda'
        if torch.cuda.is_available()
        else
        'cpu'

    )



    x=[]
    t=[]



    for n in nodes:

        x.append(
            G.nodes[n]['features']
        )

        t.append(
            G.nodes[n]['time']
        )



    x=torch.tensor(

        np.array(x),

        dtype=torch.float32,

        device=device

    )



    t=torch.tensor(

        t,

        dtype=torch.float32,

        device=device

    ).unsqueeze(1)



    t=(

        t-t.mean()

    )/(

        t.std()+1e-8

    )



    x=torch.cat(

        [x,t],

        dim=1

    )



    edges=[]
    et=[]



    for u,v,a in G.edges(data=True):


        edges.append([u,v])

        et.append(
            a['edge_type']
        )


        edges.append([v,u])

        et.append(
            a['edge_type']
        )



    edge_index=torch.tensor(

        edges,

        dtype=torch.long,

        device=device

    ).t().contiguous()



    edge_type=torch.tensor(

        et,

        dtype=torch.long,

        device=device

    )



    J=compute_jaccard_fast(

        edge_index,

        G.number_of_nodes(),

        device=device

    )



    emb=train_gnn(

        x,

        edge_index,

        edge_type,

        J,

        epochs=epochs,

        lr=1e-3,

        walk_length=20,

        top_k=10,

        device=device

    )


    return {
        'embeddings': emb.detach().cpu().numpy(),
        'subjects': np.array([G.nodes[n]['subject'] for n in nodes])
    }

")





# =========================================================
# DATA
# =========================================================


data(aids)



aids_clean <- aids



# =========================================================
# CLEANING
# =========================================================


colnames(aids_clean) = c("patient","Time","death",
"CD4", "obstime","drug","gender","prevOI","AZT")

aids_clean$patient <-
    as.numeric(aids_clean$patient)


aids_clean$obstime <-
    as.numeric(aids_clean$obstime)


aids_clean$CD4 <-
    as.numeric(aids_clean$CD4)



aids_clean$Time <-
    as.numeric(aids_clean$Time)


aids_clean$death <-
    as.numeric(aids_clean$death)



aids_clean <-

    aids_clean[

        complete.cases(

            aids_clean[,c(

                "patient",

                "obstime",

                "CD4"

            )]

        ),

    ]





# =========================================================
# SURVIVAL DATA
# =========================================================


surv_data <- unique(

    aids_clean[,c(

        "patient",

        "Time",

        "death"

    )]

)



colnames(surv_data) <- c(

    "id",

    "time",

    "event"

)



surv_data$id <-
    as.numeric(surv_data$id)


surv_data$time <-
    as.numeric(surv_data$time)


surv_data$event <-
    as.numeric(surv_data$event)




# =========================================================
# METRICS
# =========================================================


safe_cindex <- function(score,surv){


if(length(unique(score))<2)

    return(NA)



out <- try(

    rcorr.cens(

        score,

        Surv(

            surv$time,

            surv$event

        )

    ),

silent=TRUE

)



if(inherits(out,"try-error"))

    return(NA)



as.numeric(out["C Index"])

}





safe_logrank <- function(cluster,surv){


df <- data.frame(

time =
surv$time,

event =
surv$event,

cluster =
as.factor(cluster)

)



if(length(unique(df$cluster))<2)

return(NA)



survdiff(

Surv(time,event)

~

cluster,

data=df

)$chisq


}




align_clusters <- function(cluster,merged){


risk <- tapply(

merged$time,

cluster,

mean,

na.rm=TRUE

)



ord <- order(risk)



map <- setNames(

seq_along(ord),

ord

)



as.numeric(

map[

as.character(cluster)

]

)

}





# =========================================================
# PARAMETERS
# =========================================================


n_iter <- 20

k_clusters <- 3



RES <- matrix(

NA,

n_iter,

4

)



colnames(RES) <- c(

"FPCA_C",

"FPCA_LR",

"GNN_C",

"GNN_LR"

)






# =========================================================
# MAIN LOOP
# =========================================================


for(ii in 1:n_iter){


cat(

"\nITERATION",

ii,

"\n"

)




# =====================================================
# FPCA
# =====================================================


tmp <- data.frame(

id = aids_clean$patient,

time = aids_clean$obstime,

y = aids_clean$CD4

)



Ly <- split(

tmp$y,

tmp$id

)


Lt <- split(

tmp$time,

tmp$id

)



fp <- try(

FPCA(

Ly=Ly,

Lt=Lt,

optns=list(

dataType="Sparse"

)

),

silent=TRUE

)



if(inherits(fp,"try-error"))

next



scores <- fp$xiEst



rownames(scores) <-

names(Ly)



X_fpca <- scale(scores)



ids_fpca <- as.numeric(

rownames(X_fpca)

)



surv_fpca <- surv_data[

surv_data$id %in% ids_fpca,

]



cl_fpca <- kmeans(

X_fpca,

centers=k_clusters,

nstart=100

)$cluster



merged_fpca <- merge(

surv_fpca,

data.frame(

id=ids_fpca,

cluster=cl_fpca

),

by="id"

)



cl_fpca_aligned <-

align_clusters(

merged_fpca$cluster,

merged_fpca

)



c_fpca <-

safe_cindex(

cl_fpca_aligned,

merged_fpca

)



lr_fpca <-

safe_logrank(

cl_fpca_aligned,

merged_fpca

)






# =====================================================
# RANKWALK GNN INPUT
# =====================================================



Longdat2 <- data.frame(


subject =

aids_clean$patient,


time =

aids_clean$obstime,


outcome =

1,


y =

aids_clean$CD4


)


Longdat2_hack <- data.frame(


subject =

aids_clean$patient,


time =

aids_clean$obstime,


outcome =

2,


y =

aids_clean$CD4


)

# HACK
Longdat2 = rbind(Longdat2, Longdat2_hack)



print(head(Longdat2))



res <- py$run_rankwalk_gnn(

Longdat2,

100L

)



emb <- res$embeddings


sub <- as.numeric(

res$subjects

)




subjects <- sort(unique(sub))



feat <- lapply(

subjects,

function(id){

idx <- which(sub==id)


colMeans(

emb[idx,,drop=FALSE]

)

}

)



feat <- do.call(

rbind,

feat

)



rownames(feat) <- subjects



feat <- scale(feat)




cl_gnn <- kmeans(

feat,

centers=k_clusters,

nstart=100

)$cluster



merged_gnn <- merge(

surv_data,

data.frame(

id=subjects,

cluster=cl_gnn

),

by="id"

)



cl_gnn_aligned <-

align_clusters(

merged_gnn$cluster,

merged_gnn

)



c_gnn <-

safe_cindex(

cl_gnn_aligned,

merged_gnn

)



lr_gnn <-

safe_logrank(

cl_gnn_aligned,

merged_gnn

)




RES[ii,] <- c(

c_fpca,

lr_fpca,

c_gnn,

lr_gnn

)



print(RES)


}






# =========================================================
# RESULTS
# =========================================================


print(

colMeans(

RES,

na.rm=TRUE

)

)



print(

apply(

RES,

2,

sd,

na.rm=TRUE

)

)






# =========================================================
# BOXPLOTS
# =========================================================


df_C <- data.frame(

value=c(

RES[,1],

RES[,3]

),

method=rep(

c(

"FPCA",

"GNN"

),

each=n_iter

)

)




df_LR <- data.frame(

value=c(

RES[,2],

RES[,4]

),

method=rep(

c(

"FPCA",

"GNN"

),

each=n_iter

)

)




p1 <- ggplot(

df_C,

aes(

x=method,

y=value

)

)+

geom_boxplot()+

theme_minimal()+

labs(

title="C-index FPCA vs GNN"

)





p2 <- ggplot(

df_LR,

aes(

x=method,

y=value

)

)+

geom_boxplot()+

theme_minimal()+

labs(

title="Log-rank FPCA vs GNN"

)





grid.arrange(

p1,

p2,

ncol=2

)