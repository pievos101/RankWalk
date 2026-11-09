# ======================================================================
# INDUCTIVE / PROGRESSIVE longTAPIO
#
# PCA selection:
#
#   pca_selection = "first"
#       -> always use PC1
#
#   pca_selection = "random_weighted"
#       -> randomly select PC according to explained variance
#
# DEFAULT:
#   pca_selection = "first"
#
# ======================================================================


# ======================================================================
# ASSOCIATION MATRIX
# ======================================================================

.longTAPIO_association <- function(cl) {

    outer(
        cl,
        cl,
        FUN = function(a, b) {
            as.numeric(a == b)
        }
    )
}


# ======================================================================
# PREPARE LONGITUDINAL DATA
# ======================================================================

.longTAPIO_prepare_data <- function(
    DATA,
    user_id
) {

    DATA <- as.matrix(DATA)


    if(nrow(DATA) != length(user_id)) {

        stop(
            "length(user_id) must equal nrow(DATA)."
        )
    }


    patients <- sort(
        unique(user_id)
    )


    visit_counts <- table(
        user_id
    )


    if(
        length(
            unique(
                as.integer(
                    visit_counts
                )
            )
        ) != 1
    ) {

        stop(
            paste0(
                "Current implementation requires ",
                "equal numbers of visits per patient."
            )
        )
    }


    n_visits <- as.integer(
        visit_counts[1]
    )


    # ------------------------------------------------------------------
    # Patient-wise ordering.
    #
    # Existing within-patient row order is retained.
    # ------------------------------------------------------------------

    row_order <- unlist(

        lapply(

            patients,

            function(pid) {

                which(
                    user_id == pid
                )
            }
        ),

        use.names = FALSE
    )


    DATA <- DATA[
        row_order,
        ,
        drop = FALSE
    ]


    user_id <- user_id[
        row_order
    ]


    return(

        list(

            DATA =
                DATA,

            user_id =
                user_id,

            patients =
                patients,

            n_visits =
                n_visits
        )
    )
}


# ======================================================================
# EXTRACT FIRST H VISITS
# ======================================================================

.longTAPIO_extract_prefix <- function(
    DATA,
    user_id,
    visits
) {

    DATA <- as.matrix(DATA)


    patients <- sort(
        unique(user_id)
    )


    rows <- integer(0)


    for(pid in patients) {

        ids <- which(
            user_id == pid
        )


        if(length(ids) < visits) {

            stop(
                paste0(
                    "Patient ",
                    pid,
                    " has fewer than ",
                    visits,
                    " visits."
                )
            )
        }


        rows <- c(
            rows,
            ids[
                seq_len(visits)
            ]
        )
    }


    X <- DATA[
        rows,
        ,
        drop = FALSE
    ]


    uid <- rep(
        seq_along(patients),
        each = visits
    )


    return(

        list(

            DATA =
                X,

            user_id =
                uid,

            patients =
                patients
        )
    )
}


# ======================================================================
# FIT
# ======================================================================

longTAPIO_inductive <- function(
    DATA,
    user_id,
    k = 4,
    n_features = NULL,
    n_trees = 500,
    levels = 4,
    method = "ward.D2",
    scale = TRUE,
    replace = TRUE,
    pca_selection = c(
        "first",
        "random_weighted",
        "random_weighted_95"
    )
) {

    # ==================================================================
    # PCA SELECTION
    # ==================================================================

    pca_selection <- match.arg(
        pca_selection
    )

    #print(pca_selection)

    DATA <- as.matrix(DATA)


    if(!is.numeric(DATA)) {

        stop(
            "DATA must contain numeric features."
        )
    }


    if(anyNA(DATA)) {

        stop(
            "Missing values are currently not supported."
        )
    }


    tmp <- .longTAPIO_prepare_data(
        DATA,
        user_id
    )


    DATA <- tmp$DATA

    user_id <- tmp$user_id

    patients <- tmp$patients

    n_visits <- tmp$n_visits


    n_patients <- length(
        patients
    )


    p <- ncol(DATA)


    if(is.na(n_features)) {

        n_features <- max(
            1L,
            floor(sqrt(p))
        )
    }


    if(!replace && n_features > p) {

        stop(
            paste0(
                "n_features cannot exceed the number of features ",
                "when replace = FALSE."
            )
        )
    }


    # ==================================================================
    # FEATURE NAMES
    # ==================================================================

    feature_names <- colnames(DATA)


    if(is.null(feature_names)) {

        feature_names <- paste0(
            "feature_",
            seq_len(p)
        )


        colnames(DATA) <-
            feature_names
    }


    # ==================================================================
    # STORAGE
    # ==================================================================

    AFF <- matrix(
        0,
        n_patients,
        n_patients
    )


    TREES <- vector(
        "list",
        n_trees
    )


    # ==================================================================
    # TREE ENSEMBLE
    # ==================================================================

    for(b in seq_len(
        n_trees
    )) {

        # --------------------------------------------------------------
        # Random feature sampling
        # --------------------------------------------------------------

        ids <- sample(
            seq_len(p),
            n_features,
            replace = replace
        )


        DATA_s <- DATA[
            ,
            ids,
            drop = FALSE
        ]


        # --------------------------------------------------------------
        # PCA -- TRAINING DATA ONLY
        # --------------------------------------------------------------

        pca <- prcomp(
            DATA_s,
            center = TRUE,
            scale. = scale
        )


        eigenvalues <-
            pca$sdev^2


        # ==============================================================
        # SELECT PRINCIPAL COMPONENT
        #
        # first:
        #   deterministic PC1
        #
        # random_weighted:
        #   random PC from all PCs with probability proportional
        #   to explained variance
        #
        # random_weighted_95:
        #   retain the smallest set of leading PCs explaining at least
        #   95% of total variance, then randomly select one PC from this
        #   set with probability proportional to explained variance
        # ==============================================================

        if(
            pca_selection == "first"
        ) {

            selected_pc <- 1L

        } else if(
            pca_selection == "random_weighted"
        ) {

            prob <-
                eigenvalues /
                sum(
                    eigenvalues
                )

            selected_pc <- sample(
                seq_along(
                    prob
                ),
                size = 1,
                prob = prob
            )

        } else if(
            pca_selection == "random_weighted_95"
        ) {

            # Explained variance ratio
            explained_variance <-
                eigenvalues /
                sum(
                    eigenvalues
                )

            # Smallest number of leading PCs explaining >= 95%
            n_pc <- which(
                cumsum(
                    explained_variance
                ) >= 0.95
            )[1]

            eligible_pc <- seq_len(
                n_pc
            )

            # Variance-weighted probabilities within retained PCs
            prob <-
                eigenvalues[
                    eligible_pc
                ] /
                sum(
                    eigenvalues[
                        eligible_pc
                    ]
                )

            selected_pc <- sample(
                eligible_pc,
                size = 1,
                prob = prob
            )

        } else {

            stop(
                "Unknown pca_selection: ",
                pca_selection
            )
        }


        scores <- drop(
            pca$x[
                ,
                selected_pc
            ]
        )


        # ==============================================================
        # FEATURE CONTRIBUTION
        # ==============================================================

        pc_cor <-
            pca$rotation[
                ,
                selected_pc
            ] *
            pca$sdev[
                selected_pc
            ]


        feature_contribution <-
            pc_cor^2


        contribution_sum <-
            sum(
                feature_contribution
            )


        if(
            is.finite(
                contribution_sum
            ) &&
            contribution_sum > 0
        ) {

            feature_contribution <-
                feature_contribution /
                contribution_sum

        } else {

            feature_contribution[] <- 0
        }


        # --------------------------------------------------------------
        # Original feature IDs.
        #
        # Duplicates are intentional when replace = TRUE.
        # --------------------------------------------------------------

        names(
            feature_contribution
        ) <- as.character(
            ids
        )


        # ==============================================================
        # COMPLETE PATIENT TRAJECTORIES
        # ==============================================================

        trajectories <- matrix(
            scores,
            nrow = n_patients,
            ncol = n_visits,
            byrow = TRUE
        )


        rownames(
            trajectories
        ) <- as.character(
            patients
        )


        # ==============================================================
        # HIERARCHICAL CLUSTERING
        # ==============================================================

        hc <- fastcluster::hclust(
            dist(
                trajectories
            ),
            method = method
        )


        LEVEL_CLUSTERS <- vector(
            "list",
            levels
        )


        LEVEL_CENTROIDS <- vector(
            "list",
            levels
        )


        # ==============================================================
        # HIERARCHY LEVELS
        # ==============================================================

        for(l in seq_len(
            levels
        )) {

            cl <- cutree(
                hc,
                k = l + 1
            )


            LEVEL_CLUSTERS[[l]] <-
                cl


            cluster_ids <- sort(
                unique(
                    cl
                )
            )


            centroids <- matrix(
                NA_real_,
                nrow = length(
                    cluster_ids
                ),
                ncol = n_visits
            )


            rownames(
                centroids
            ) <- as.character(
                cluster_ids
            )


            for(cc in seq_along(
                cluster_ids
            )) {

                cid <-
                    cluster_ids[cc]


                members <- which(
                    cl == cid
                )


                centroids[
                    cc,
                ] <- colMeans(
                    trajectories[
                        members,
                        ,
                        drop = FALSE
                    ]
                )
            }


            LEVEL_CENTROIDS[[l]] <-
                centroids


            AFF <- AFF +
                .longTAPIO_association(
                    cl
                )
        }


        # ==============================================================
        # STORE FROZEN TREE
        # ==============================================================

        TREES[[b]] <- list(

            feature_ids =
                ids,

            pca_center =
                pca$center,

            pca_scale =
                pca$scale,

            pca_rotation =
                pca$rotation[
                    ,
                    selected_pc
                ],

            selected_pc =
                selected_pc,

            selected_eigenvalue =
                eigenvalues[
                    selected_pc
                ],

            feature_contribution =
                feature_contribution,

            train_trajectories =
                trajectories,

            level_clusters =
                LEVEL_CLUSTERS,

            level_centroids =
                LEVEL_CENTROIDS
        )
    }


    # ==================================================================
    # NORMALIZED TRAINING AFFINITY
    # ==================================================================

    AFF <- AFF /
        (
            n_trees *
            levels
        )


    DIST <- 1 - AFF

    diag(
        DIST
    ) <- 0


    # ==================================================================
    # FINAL TRAINING CLUSTERING
    # ==================================================================

    final_hclust <- fastcluster::hclust(
        as.dist(
            DIST
        ),
        method = method
    )


    final_clusters <- cutree(
        final_hclust,
        k = k
    )


    names(
        final_clusters
    ) <- as.character(
        patients
    )


    # ==================================================================
    # MODEL
    # ==================================================================

    model <- list(

        call =
            match.call(),

        trees =
            TREES,

        cluster =
            final_clusters,

        train_clusters =
            final_clusters,

        patients =
            patients,

        train_patients =
            patients,

        affinity =
            AFF,

        distance =
            DIST,

        final_hclust =
            final_hclust,

        k =
            k,

        levels =
            levels,

        n_trees =
            n_trees,

        n_features =
            n_features,

        n_features_total =
            p,

        n_visits =
            n_visits,

        feature_names =
            feature_names,

        method =
            method,

        scale =
            scale,

        replace =
            replace,

        pca_selection =
            pca_selection
    )


    class(
        model
    ) <- "inductiveLongTAPIO"


    return(
        model
    )
}


# ======================================================================
# PROJECT PREFIX THROUGH ONE FROZEN TREE
# ======================================================================

.longTAPIO_project_prefix <- function(
    tree,
    DATA,
    n_patients,
    visits
) {

    DATA <- as.matrix(
        DATA
    )


    Xs <- DATA[
        ,
        tree$feature_ids,
        drop = FALSE
    ]


    # ------------------------------------------------------------------
    # Frozen PCA center
    # ------------------------------------------------------------------

    Xs <- sweep(
        Xs,
        2,
        tree$pca_center,
        "-"
    )


    # ------------------------------------------------------------------
    # Frozen PCA scale
    # ------------------------------------------------------------------

    if(
        !is.null(
            tree$pca_scale
        ) &&
        !identical(
            tree$pca_scale,
            FALSE
        )
    ) {

        Xs <- sweep(
            Xs,
            2,
            tree$pca_scale,
            "/"
        )
    }


    # ------------------------------------------------------------------
    # Frozen selected PC
    # ------------------------------------------------------------------

    scores <- drop(
        Xs %*%
        tree$pca_rotation
    )


    # ------------------------------------------------------------------
    # Patient x prefix trajectory
    # ------------------------------------------------------------------

    trajectories <- matrix(
        scores,
        nrow = n_patients,
        ncol = visits,
        byrow = TRUE
    )


    return(
        trajectories
    )
}


# ======================================================================
# ONE TREE:
# PREFIX -> TRAINING AFFINITY
# ======================================================================

.longTAPIO_tree_prefix_affinity <- function(
    tree,
    new_trajectories,
    visits
) {

    n_new <- nrow(
        new_trajectories
    )


    n_train <- nrow(
        tree$train_trajectories
    )


    A <- matrix(
        0,
        n_new,
        n_train
    )


    for(l in seq_along(
        tree$level_clusters
    )) {

        train_cl <-
            tree$level_clusters[[l]]


        full_centroids <-
            tree$level_centroids[[l]]


        # --------------------------------------------------------------
        # Prefix of frozen full-trajectory prototypes
        # --------------------------------------------------------------

        centroids <- full_centroids[
            ,
            seq_len(
                visits
            ),
            drop = FALSE
        ]


        cluster_ids <-
            as.integer(
                rownames(
                    centroids
                )
            )


        # --------------------------------------------------------------
        # Nearest frozen prototype
        # --------------------------------------------------------------

        for(i in seq_len(
            n_new
        )) {

            z <- new_trajectories[
                i,
                ,
                drop = TRUE
            ]


            distances <- apply(

                centroids,

                1,

                function(mu) {

                    sum(
                        (z - mu)^2
                    )
                }
            )


            new_cl <-
                cluster_ids[
                    which.min(
                        distances
                    )
                ]


            A[
                i,
            ] <- A[
                i,
            ] +
                as.numeric(
                    train_cl ==
                    new_cl
                )
        }
    }


    return(
        A
    )
}


# ======================================================================
# ENSEMBLE PREFIX AFFINITY
# ======================================================================

.longTAPIO_predict_affinity <- function(
    object,
    DATA,
    user_id,
    visits
) {

    prefix <- .longTAPIO_extract_prefix(
        DATA = DATA,
        user_id = user_id,
        visits = visits
    )


    n_new <- length(
        prefix$patients
    )


    n_train <- length(
        object$train_clusters
    )


    AFF_NEW <- matrix(
        0,
        n_new,
        n_train
    )


    for(b in seq_len(
        object$n_trees
    )) {

        trajectories <-
            .longTAPIO_project_prefix(

                tree =
                    object$trees[[b]],

                DATA =
                    prefix$DATA,

                n_patients =
                    n_new,

                visits =
                    visits
            )


        AFF_NEW <- AFF_NEW +
            .longTAPIO_tree_prefix_affinity(

                tree =
                    object$trees[[b]],

                new_trajectories =
                    trajectories,

                visits =
                    visits
            )
    }


    AFF_NEW <- AFF_NEW /
        (
            object$n_trees *
            object$levels
        )


    return(

        list(

            affinity =
                AFF_NEW,

            patients =
                prefix$patients
        )
    )
}


# ======================================================================
# S3 PREDICT METHOD
# ======================================================================

predict.inductiveLongTAPIO <- function(
    object,
    newdata,
    user_id,
    visits = object$n_visits,
    ...
) {

    DATA <- as.matrix(
        newdata
    )


    if(!is.numeric(DATA)) {

        stop(
            "newdata must contain numeric features."
        )
    }


    if(anyNA(DATA)) {

        stop(
            "Missing values are currently not supported."
        )
    }


    if(
        nrow(DATA) !=
        length(
            user_id
        )
    ) {

        stop(
            "length(user_id) must equal nrow(newdata)."
        )
    }


    if(
        ncol(DATA) !=
        object$n_features_total
    ) {

        stop(
            paste0(
                "newdata contains ",
                ncol(DATA),
                " features; model expects ",
                object$n_features_total,
                "."
            )
        )
    }


    if(
        !is.null(
            colnames(
                DATA
            )
        ) &&
        !identical(
            colnames(
                DATA
            ),
            object$feature_names
        )
    ) {

        stop(
            "Feature names/order do not match training data."
        )
    }


    if(
        visits < 1 ||
        visits > object$n_visits
    ) {

        stop(
            paste0(
                "visits must be between 1 and ",
                object$n_visits,
                "."
            )
        )
    }


    tmp <- .longTAPIO_predict_affinity(

        object =
            object,

        DATA =
            DATA,

        user_id =
            user_id,

        visits =
            visits
    )


    Anew <- tmp$affinity


    n_new <- nrow(
        Anew
    )


    # ==================================================================
    # AFFINITY TO FINAL REFERENCE CLUSTERS
    # ==================================================================

    cluster_scores <- matrix(
        NA_real_,
        n_new,
        object$k
    )


    for(k_id in seq_len(
        object$k
    )) {

        members <- which(
            object$train_clusters ==
            k_id
        )


        cluster_scores[
            ,
            k_id
        ] <- rowMeans(
            Anew[
                ,
                members,
                drop = FALSE
            ]
        )
    }


    # ==================================================================
    # ASSIGNMENT
    # ==================================================================

    predicted <- max.col(
        cluster_scores,
        ties.method = "first"
    )


    # ==================================================================
    # ASSIGNMENT MARGIN
    # ==================================================================

    margin <- apply(

        cluster_scores,

        1,

        function(x) {

            sx <- sort(
                x,
                decreasing = TRUE
            )


            if(length(sx) < 2) {

                return(
                    NA_real_
                )
            }


            sx[1] - sx[2]
        }
    )


    return(

        list(

            cluster =
                predicted,

            cluster_scores =
                cluster_scores,

            margin =
                margin,

            # Compatibility alias.
            # Note: this is an assignment margin, not a probability.
            confidence =
                margin,

            affinity =
                Anew,

            patients =
                tmp$patients,

            visits =
                visits
        )
    )
}