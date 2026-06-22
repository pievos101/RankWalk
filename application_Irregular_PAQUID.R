# =========================================================
# LIBRARIES
# =========================================================

library(MASS)
library(aricode)
library(fda)
library(survival)
library(survcomp)


# =========================================================
# LOAD DATA
# =========================================================

load("paquid.rda")

paquid <- as.data.frame(paquid)



# =========================================================
# KEEP ONLY PATIENTS WITH > 3 VISITS
# =========================================================

visit_counts <- table(paquid$ID)

keep_ids <- names(
  visit_counts[visit_counts > 3]
)

paquid <- paquid[
  paquid$ID %in% keep_ids,
]



# =========================================================
# VARIABLES
# =========================================================

vars <- c(
  "MMSE",
  "BVRT",
  "IST",
  "CESD"
)



# =========================================================
# NUMERIC CONVERSION
# =========================================================

for(v in vars){

paquid[[v]] <-
suppressWarnings(
as.numeric(
as.character(
paquid[[v]]
)
)
)

}



paquid$age <- as.numeric(paquid$age)

paquid$ID <- as.numeric(paquid$ID)

paquid$dem <- as.numeric(paquid$dem)



# =========================================================
# TIME VARIABLE
# =========================================================

paquid <- paquid[
order(paquid$ID, paquid$age),
]


paquid$time <- paquid$age



# =========================================================
# LOG TRANSFORM
# =========================================================

for(v in vars){

paquid[[v]] <- log1p(
paquid[[v]]
)

}



# =========================================================
# CLEAN DATA
# =========================================================


Ymat <- data.matrix(
paquid[,vars]
)


keep <- rowSums(
is.finite(Ymat)
)>0



paquid_clean <- paquid[keep,]


paquid_clean[,vars] <-
Ymat[keep,]



paquid_clean <-
paquid_clean[
is.finite(paquid_clean$time),
]





# =========================================================
# SURVIVAL DATA
# =========================================================


surv_df <-

do.call(

rbind,

lapply(

split(paquid_clean,
      paquid_clean$ID),

function(df){


df <- df[
order(df$age),
]


event <-
any(
df$dem==1,
na.rm=TRUE
)



if(event){

event_time <-
min(
df$age[df$dem==1],
na.rm=TRUE
)

}else{

event_time <-
max(
df$age,
na.rm=TRUE
)

}



baseline_age <-
min(
df$age,
na.rm=TRUE
)



data.frame(

ID =
unique(df$ID),

time =
event_time-baseline_age,

event =
as.numeric(event)

)

}

)

)



surv_df <-
surv_df[
is.finite(surv_df$time),
]





# =========================================================
# GNN PYTHON SETUP
# =========================================================


Sys.setenv(
RETICULATE_PYTHON =
path.expand("~/rankwalk-venv/bin/python")
)



use_python(
Sys.getenv("RETICULATE_PYTHON"),
required=TRUE
)




py_run_string("

import numpy as np
import pandas as pd
import torch

from rankwalk import build_temporal_graph_grid
from rankwalk import train_gnn
from rankwalk import compute_jaccard_fast



def run_rankwalk_gnn(df,epochs=100):


    df=pd.DataFrame(df)


    G,_=build_temporal_graph_grid(
        df,
        k_similarity=10,
        n_bins=5,
        overlap=0.8
    )


    nodes=list(G.nodes())


    device=torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
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
        walk_length=5,
        top_k=5,
        device=device
    )



    return {

    'embeddings':
    emb.detach().cpu().numpy(),

    'subjects':
    np.array(
    [
    G.nodes[n]['subject']
    for n in nodes
    ])

    }

")





# =========================================================
# LONG FORMAT FOR GNN
# =========================================================


Longdat_list <- list()


for(v in vars){


tmp <- paquid_clean[

is.finite(paquid_clean[[v]]) &
is.finite(paquid_clean$time),

c("ID","time",v)

]


colnames(tmp)[3]="y"



Longdat_list[[v]] <-

data.frame(

subject=tmp$ID,

time=tmp$time,

outcome=v,

y=tmp$y

)

}



Longdat2 <-

do.call(
rbind,
Longdat_list
)



# =========================================================
# RESULT MATRIX
# =========================================================


n_iter <- 30



RES <- matrix(
NA,
nrow=n_iter,
ncol=4
)



colnames(RES)<-

c(

"FPCA_Cindex",

"FPCA_Chisq",

"GNN_Cindex",

"GNN_Chisq"

)




# =========================================================
# 30 ITERATIONS
# =========================================================


for(iter in 1:n_iter){


cat(
"\nITERATION",
iter,
"\n"
)



set.seed(iter)



# =========================================================
# FPCA
# =========================================================


fpca_features <- list()



for(v in vars){


tmp <- paquid_clean[

is.finite(paquid_clean[[v]]) &
is.finite(paquid_clean$time),

c("ID","time",v)

]


colnames(tmp)[3]="y"



Ly <- split(
tmp$y,
tmp$ID
)



Lt <- split(
tmp$time,
tmp$ID
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



if(is.null(dim(scores)))

scores <-
matrix(scores,ncol=1)



rownames(scores)<-names(Ly)



scores[!is.finite(scores)]<-0



fpca_features[[v]]<-scores


}





ids_all <- sort(
unique(paquid_clean$ID)
)



X_fpca <- NULL



for(v in names(fpca_features)){


S <- fpca_features[[v]]


tmp <- matrix(

0,

nrow=length(ids_all),

ncol=ncol(S)

)


rownames(tmp)<-ids_all



common <-
intersect(
rownames(S),
ids_all
)



tmp[
match(common,ids_all),
] <-
S[common,,drop=FALSE]



X_fpca <-

if(is.null(X_fpca))

tmp

else

cbind(
X_fpca,
tmp
)

}



X_fpca[!is.finite(X_fpca)]<-0


X_fpca<-scale(X_fpca)



cl_fpca <- kmeans(

X_fpca,

centers=3,

nstart=50

)$cluster



fpca_df <- data.frame(

ID=as.numeric(
rownames(X_fpca)
),

cluster=cl_fpca

)




fpca_eval <- merge(

fpca_df,

surv_df,

by="ID"

)



fpca_eval$cluster <-
factor(fpca_eval$cluster)



lr_fpca <- survdiff(

Surv(time,event)~cluster,

data=fpca_eval

)



fpca_chisq <- lr_fpca$chisq



cox_fpca <- coxph(

Surv(time,event)~cluster,

data=fpca_eval

)



fpca_cindex <-

cox_fpca$concordance[6]





# =========================================================
# GNN
# =========================================================



res <-

py$run_rankwalk_gnn(
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

function(g){

idx <- which(sub==g)

E <- emb[idx,,drop=FALSE]

as.numeric(t(E))

}

)



max_len <-
max(
sapply(feat,length)
)



feat <-

t(

sapply(

feat,

function(x)

c(
x,
rep(0,max_len-length(x))
)

)

)



rownames(feat)<-subjects


feat[!is.finite(feat)]<-0


feat<-scale(feat)



cl_gnn <- kmeans(

feat,

centers=3,

nstart=50

)$cluster



gnn_df <- data.frame(

ID=as.numeric(names(cl_gnn)),

cluster=cl_gnn

)




gnn_eval <- merge(

gnn_df,

surv_df,

by="ID"

)



gnn_eval$cluster <-

factor(
gnn_eval$cluster
)



lr_gnn <- survdiff(

Surv(time,event)~cluster,

data=gnn_eval

)



gnn_chisq <- lr_gnn$chisq



cox_gnn <- coxph(

Surv(time,event)~cluster,

data=gnn_eval

)



gnn_cindex <-

cox_gnn$concordance[6]





# =========================================================
# SAVE RESULTS
# =========================================================


RES[iter,] <-

c(

fpca_cindex,

fpca_chisq,

gnn_cindex,

gnn_chisq

)



cat(

"FPCA C:",
round(fpca_cindex,3),

"GNN C:",
round(gnn_cindex,3),

"\n"

)


print(RES)


}


library(ggplot2)
library(gridExtra)

# =========================================================
# RES must already exist
# columns:
# FPCA_Cindex FPCA_Chisq GNN_Cindex GNN_Chisq
# =========================================================

RES <- as.data.frame(RES)

# =========================================================
# CREATE DATA FRAMES (base R only)
# =========================================================

df_cindex <- data.frame(
  FPCA = RES$FPCA_Cindex,
  GNN  = RES$GNN_Cindex
)

df_chisq <- data.frame(
  FPCA = RES$FPCA_Chisq,
  GNN  = RES$GNN_Chisq
)

# =========================================================
# SUMMARY FUNCTION (mean + sd) - base R
# =========================================================

summarise_df <- function(df) {
  data.frame(
    Method = c("FPCA", "GNN"),
    Mean = c(mean(df$FPCA, na.rm = TRUE),
             mean(df$GNN, na.rm = TRUE)),
    SD = c(sd(df$FPCA, na.rm = TRUE),
           sd(df$GNN, na.rm = TRUE))
  )
}

sum_cindex <- summarise_df(df_cindex)
sum_chisq  <- summarise_df(df_chisq)

# =========================================================
# PLOT 1: C-index
# =========================================================

p1 <- ggplot(sum_cindex, aes(x = Method, y = Mean, fill = Method)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(
    aes(ymin = Mean - SD, ymax = Mean + SD),
    width = 0.2
  ) +
  ylim(0, 1) +
  ggtitle("C-index (FPCA vs GNN)") +
  ylab("C-index") +
  theme_minimal() +
  theme(legend.position = "none")

# =========================================================
# PLOT 2: Chi-square
# =========================================================

p2 <- ggplot(sum_chisq, aes(x = Method, y = Mean, fill = Method)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(
    aes(ymin = Mean - SD, ymax = Mean + SD),
    width = 0.2
  ) +
  ggtitle("Log-rank Chi-square (FPCA vs GNN)") +
  ylab("Chi-square") +
  theme_minimal() +
  theme(legend.position = "none")

# =========================================================
# SIDE-BY-SIDE PLOTS (gridExtra)
# =========================================================

grid.arrange(p1, p2, ncol = 2)





