# ======================================================================
# FEATURE IMPORTANCE FOR INDUCTIVE / PROGRESSIVE longTAPIO
# ======================================================================


importance_longTAPIO_inductive <- function(
    res,
    newdata = NULL,
    user_id = NULL,
    normalize = TRUE
) {

    # ==================================================================
    # CHECK MODEL
    # ==================================================================

    if(!inherits(
        res,
        "inductiveLongTAPIO"
    )) {

        stop(
            "res must be returned by longTAPIO_inductive()."
        )
    }


    K <- res$k

    P <- res$n_features_total

    B <- res$n_trees

    T <- res$n_visits


    n_train <- length(
        res$train_clusters
    )


    # ==================================================================
    # TREE FEATURE CONTRIBUTIONS IN ORIGINAL P-DIMENSIONAL SPACE
    # ==================================================================

    TREE_FEATURE_IMP <- matrix(
        0,
        B,
        P
    )


    colnames(
        TREE_FEATURE_IMP
    ) <- res$feature_names


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
                    "Refit with the revised longTAPIO_inductive()."
                )
            )
        }


        ids <-
            tree$feature_ids


        contribution <-
            tree$feature_contribution


        # --------------------------------------------------------------
        # Duplicate sampled features are summed.
        # --------------------------------------------------------------

        for(j in seq_along(ids)) {

            TREE_FEATURE_IMP[
                b,
                ids[j]
            ] <-
                TREE_FEATURE_IMP[
                    b,
                    ids[j]
                ] +
                contribution[j]
        }
    }


    # ==================================================================
    # GLOBAL REFERENCE-CLUSTER IMPORTANCE
    # ==================================================================

    GLOBAL <- matrix(
        0,
        K,
        P
    )


    rownames(GLOBAL) <-
        paste0(
            "cluster_",
            seq_len(K)
        )


    colnames(GLOBAL) <-
        res$feature_names


    for(b in seq_len(B)) {

        tree <- res$trees[[b]]


        # --------------------------------------------------------------
        # Tree-specific training affinity
        # --------------------------------------------------------------

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


        # --------------------------------------------------------------
        # Cluster support
        # --------------------------------------------------------------

        for(k_id in seq_len(K)) {

            members <- which(
                res$train_clusters ==
                k_id
            )


            if(length(members) == 0) {
                next
            }


            cluster_support <- mean(
                A_b[
                    members,
                    members,
                    drop = FALSE
                ]
            )


            GLOBAL[
                k_id,
            ] <- GLOBAL[
                k_id,
            ] +
                cluster_support *
                TREE_FEATURE_IMP[
                    b,
                ]
        }
    }


    GLOBAL <- GLOBAL /
        B


    # ==================================================================
    # NORMALIZE GLOBAL IMPORTANCE
    # ==================================================================

    if(normalize) {

        mx <- max(
            GLOBAL,
            na.rm = TRUE
        )


        if(
            is.finite(mx) &&
            mx > 0
        ) {

            GLOBAL <-
                GLOBAL /
                mx
        }
    }


    # ==================================================================
    # GLOBAL ONLY
    # ==================================================================

    if(is.null(newdata)) {

        return(

            list(
                global =
                    GLOBAL
            )
        )
    }


    # ==================================================================
    # CHECK NEW DATA
    # ==================================================================

    if(is.null(user_id)) {

        stop(
            "user_id must be supplied when newdata is supplied."
        )
    }


    DATA <- as.matrix(
        newdata
    )


    if(anyNA(DATA)) {

        stop(
            "Missing values are currently not supported."
        )
    }


    if(
        nrow(DATA) !=
        length(user_id)
    ) {

        stop(
            "length(user_id) must equal nrow(newdata)."
        )
    }


    if(
        ncol(DATA) !=
        P
    ) {

        stop(
            paste0(
                "newdata contains ",
                ncol(DATA),
                " features; model expects ",
                P,
                "."
            )
        )
    }


    if(
        !is.null(colnames(DATA)) &&
        !identical(
            colnames(DATA),
            res$feature_names
        )
    ) {

        stop(
            "Feature names/order do not match training data."
        )
    }


    patients <- sort(
        unique(user_id)
    )


    N <- length(
        patients
    )


    visit_counts <- table(
        user_id
    )


    if(
        any(
            visit_counts[
                as.character(patients)
            ] < T
        )
    ) {

        stop(
            paste0(
                "Every new patient must contain at least ",
                T,
                " visits."
            )
        )
    }


    # ==================================================================
    # STORAGE
    #
    # Patient x Feature x Visit
    # ==================================================================

    PATIENT_IMP <- array(

        0,

        dim = c(
            N,
            P,
            T
        ),

        dimnames = list(

            patient =
                as.character(
                    patients
                ),

            feature =
                res$feature_names,

            visit =
                paste0(
                    "visit_",
                    seq_len(T)
                )
        )
    )


    PREDICTED_CLUSTER <- matrix(

        NA_integer_,

        N,

        T,

        dimnames = list(

            patient =
                as.character(
                    patients
                ),

            visit =
                paste0(
                    "visit_",
                    seq_len(T)
                )
        )
    )


    # ==================================================================
    # PROGRESSIVE PREFIXES
    # ==================================================================

    for(H in seq_len(T)) {

        # --------------------------------------------------------------
        # Extract prefix exactly as prediction does.
        # --------------------------------------------------------------

        prefix <- .longTAPIO_extract_prefix(

            DATA =
                DATA,

            user_id =
                user_id,

            visits =
                H
        )


        # --------------------------------------------------------------
        # Tree-specific affinities
        # --------------------------------------------------------------

        TREE_AFFINITY <- vector(
            "list",
            B
        )


        A_TOTAL <- matrix(
            0,
            N,
            n_train
        )


        for(b in seq_len(B)) {

            tree <- res$trees[[b]]


            trajectories <-
                .longTAPIO_project_prefix(

                    tree =
                        tree,

                    DATA =
                        prefix$DATA,

                    n_patients =
                        N,

                    visits =
                        H
                )


            A_b <-
                .longTAPIO_tree_prefix_affinity(

                    tree =
                        tree,

                    new_trajectories =
                        trajectories,

                    visits =
                        H
                )


            # ----------------------------------------------------------
            # Normalize each tree over hierarchy levels
            # ----------------------------------------------------------

            A_b <- A_b /
                res$levels


            TREE_AFFINITY[[b]] <-
                A_b


            A_TOTAL <- A_TOTAL +
                A_b
        }


        # --------------------------------------------------------------
        # Ensemble affinity
        # --------------------------------------------------------------

        A_TOTAL <- A_TOTAL /
            B


        # ==============================================================
        # CURRENT CLUSTER ASSIGNMENT
        # ==============================================================

        cluster_scores <- matrix(
            NA_real_,
            N,
            K
        )


        for(k_id in seq_len(K)) {

            members <- which(
                res$train_clusters ==
                k_id
            )


            cluster_scores[
                ,
                k_id
            ] <- rowMeans(
                A_TOTAL[
                    ,
                    members,
                    drop = FALSE
                ]
            )
        }


        current_cluster <- max.col(
            cluster_scores,
            ties.method = "first"
        )


        PREDICTED_CLUSTER[
            ,
            H
        ] <- current_cluster


        # ==============================================================
        # PATIENT-SPECIFIC IMPORTANCE
        # ==============================================================

        for(i in seq_len(N)) {

            assigned_cluster <-
                current_cluster[i]


            members <- which(
                res$train_clusters ==
                assigned_cluster
            )


            patient_imp <- numeric(P)


            for(b in seq_len(B)) {

                tree_support <- mean(

                    TREE_AFFINITY[[b]][
                        i,
                        members,
                        drop = TRUE
                    ]
                )


                patient_imp <-
                    patient_imp +
                    tree_support *
                    TREE_FEATURE_IMP[
                        b,
                    ]
            }


            patient_imp <-
                patient_imp /
                B


            # ----------------------------------------------------------
            # Patient/prefix normalization
            # ----------------------------------------------------------

            if(normalize) {

                mx <- max(
                    patient_imp,
                    na.rm = TRUE
                )


                if(
                    is.finite(mx) &&
                    mx > 0
                ) {

                    patient_imp <-
                        patient_imp /
                        mx
                }
            }


            PATIENT_IMP[
                i,
                ,
                H
            ] <- patient_imp
        }
    }


    # ==================================================================
    # FINAL FULL-TRAJECTORY ASSIGNMENT
    # ==================================================================

    FINAL_CLUSTER <-
        PREDICTED_CLUSTER[
            ,
            T
        ]


    names(FINAL_CLUSTER) <-
        as.character(
            patients
        )


    # ==================================================================
    # CLUSTER-PROGRESSIVE IMPORTANCE
    #
    # IMPORTANT:
    #
    # Patients are grouped according to FINAL cluster assignment.
    #
    # Thus:
    #
    # "For patients ultimately assigned to cluster k,
    #  how did feature importance evolve from visits 1,...,T?"
    # ==================================================================

    CLUSTER_PROGRESSIVE <- array(

        NA_real_,

        dim = c(
            K,
            P,
            T
        ),

        dimnames = list(

            cluster =
                paste0(
                    "cluster_",
                    seq_len(K)
                ),

            feature =
                res$feature_names,

            visit =
                paste0(
                    "visit_",
                    seq_len(T)
                )
        )
    )


    for(k_id in seq_len(K)) {

        ids <- which(
            FINAL_CLUSTER ==
            k_id
        )


        if(length(ids) == 0) {
            next
        }


        for(H in seq_len(T)) {

            if(length(ids) == 1) {

                CLUSTER_PROGRESSIVE[
                    k_id,
                    ,
                    H
                ] <-
                    PATIENT_IMP[
                        ids,
                        ,
                        H
                    ]

            } else {

                tmp <-
                    PATIENT_IMP[
                        ids,
                        ,
                        H,
                        drop = FALSE
                    ]


                CLUSTER_PROGRESSIVE[
                    k_id,
                    ,
                    H
                ] <-
                    apply(
                        tmp,
                        2,
                        mean,
                        na.rm = TRUE
                    )
            }
        }
    }


    # ==================================================================
    # INCREMENTAL IMPORTANCE
    # ==================================================================

    INCREMENTAL <- array(

        NA_real_,

        dim =
            dim(
                CLUSTER_PROGRESSIVE
            ),

        dimnames =
            dimnames(
                CLUSTER_PROGRESSIVE
            )
    )


    INCREMENTAL[
        ,
        ,
        1
    ] <-
        CLUSTER_PROGRESSIVE[
            ,
            ,
            1
        ]


    if(T > 1) {

        for(H in 2:T) {

            INCREMENTAL[
                ,
                ,
                H
            ] <-
                CLUSTER_PROGRESSIVE[
                    ,
                    ,
                    H
                ] -
                CLUSTER_PROGRESSIVE[
                    ,
                    ,
                    H - 1
                ]
        }
    }


    # ==================================================================
    # RESULT
    # ==================================================================

    result <- list(

        global =
            GLOBAL,

        patient =
            PATIENT_IMP,

        cluster_progressive =
            CLUSTER_PROGRESSIVE,

        incremental =
            INCREMENTAL,

        predicted_cluster =
            PREDICTED_CLUSTER,

        final_cluster =
            FINAL_CLUSTER,

        patients =
            patients,

        n_visits =
            T,

        feature_names =
            res$feature_names
    )


    class(result) <-
        "importance.inductiveLongTAPIO"


    return(result)
}