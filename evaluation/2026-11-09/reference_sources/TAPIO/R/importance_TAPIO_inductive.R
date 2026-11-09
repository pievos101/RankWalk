# ======================================================================
# FEATURE IMPORTANCE FOR INDUCTIVE TAPIO
# ======================================================================


importance_TAPIO_inductive <- function(
    res,
    normalize = TRUE
) {

    # ==================================================================
    # CHECK MODEL
    # ==================================================================

    if(!inherits(
        res,
        "inductiveTAPIO"
    )) {

        stop(
            "res must be returned by TAPIO_inductive()."
        )
    }


    K <- res$k

    P <- res$n_features_total

    B <- res$n_trees


    n_train <- length(
        res$train_clusters
    )


    # ==================================================================
    # OUTPUT
    # ==================================================================

    CLASS_IMP <- matrix(
        0,
        K,
        P
    )


    rownames(CLASS_IMP) <-
        paste0(
            "cluster_",
            seq_len(K)
        )


    colnames(CLASS_IMP) <-
        res$feature_names


    # ==================================================================
    # TREES
    # ==================================================================

    for(b in seq_len(B)) {

        tree <- res$trees[[b]]


        if(is.null(
            tree$feature_contribution
        )) {

            stop(
                paste0(
                    "Tree ",
                    b,
                    " contains no feature_contribution. ",
                    "Refit with the revised TAPIO_inductive()."
                )
            )
        }


        # ==============================================================
        # TREE-SPECIFIC TRAINING AFFINITY
        # ==============================================================

        A_b <- matrix(
            0,
            n_train,
            n_train
        )


        for(l in seq_along(
            tree$level_clusters
        )) {

            cl <-
                tree$level_clusters[[l]]


            A_b <- A_b +
                outer(
                    cl,
                    cl,
                    FUN = function(a, b) {
                        as.numeric(a == b)
                    }
                )
        }


        A_b <- A_b /
            res$levels


        # ==============================================================
        # MAP FEATURE CONTRIBUTION TO ORIGINAL FEATURE SPACE
        # ==============================================================

        F_b <- numeric(P)


        ids <-
            tree$feature_ids


        contribution <-
            tree$feature_contribution


        # --------------------------------------------------------------
        # Important:
        # add contributions if the same feature was sampled more than
        # once with replace = TRUE.
        # --------------------------------------------------------------

        for(j in seq_along(ids)) {

            F_b[
                ids[j]
            ] <- F_b[
                ids[j]
            ] +
                contribution[j]
        }


        # ==============================================================
        # CLUSTER-SPECIFIC SUPPORT
        # ==============================================================

        for(k_id in seq_len(K)) {

            members <- which(
                res$train_clusters ==
                k_id
            )


            if(length(members) == 0) {
                next
            }


            support <- mean(
                A_b[
                    members,
                    members,
                    drop = FALSE
                ]
            )


            CLASS_IMP[
                k_id,
            ] <- CLASS_IMP[
                k_id,
            ] +
                support *
                F_b
        }
    }


    # ==================================================================
    # AVERAGE OVER TREES
    # ==================================================================

    CLASS_IMP <-
        CLASS_IMP /
        B


    # ==================================================================
    # NORMALIZATION
    # ==================================================================

    if(normalize) {

        mx <- max(
            CLASS_IMP,
            na.rm = TRUE
        )


        if(
            is.finite(mx) &&
            mx > 0
        ) {

            CLASS_IMP <-
                CLASS_IMP /
                mx
        }
    }


    return(
        CLASS_IMP
    )
}