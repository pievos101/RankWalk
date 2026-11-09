# ======================================================================
# INDUCTIVE TAPIO
# ======================================================================


# ======================================================================
# ASSOCIATION MATRIX
# ======================================================================

.TAPIO_association <- function(cl) {

    outer(
        cl,
        cl,
        FUN = function(a, b) {
            as.numeric(a == b)
        }
    )
}


# ======================================================================
# FIT
# ======================================================================

TAPIO_inductive <- function(
    DATA,
    k = 4,
    n_features = NULL,
    n_trees = 500,
    levels = 10,
    method = "ward.D2",
    scale = FALSE,
    replace = TRUE
) {

    DATA <- as.matrix(DATA)

    if(!is.numeric(DATA)) {
        stop("DATA must contain numeric features.")
    }

    if(anyNA(DATA)) {
        stop("Missing values are currently not supported.")
    }

    n <- nrow(DATA)
    p <- ncol(DATA)


    if(is.null(n_features)) {
        n_features <- max(
            1L,
            floor(sqrt(p))
        )
    }


    if(!replace && n_features > p) {
        stop(
            "n_features cannot exceed the number of features ",
            "when replace = FALSE."
        )
    }


    if(k < 2 || k > n) {
        stop("Invalid number of final clusters.")
    }


    if(levels < 1 || levels + 1 > n) {
        stop("Invalid number of hierarchy levels.")
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

        colnames(DATA) <- feature_names
    }


    # ==================================================================
    # STORAGE
    # ==================================================================

    AFF <- matrix(
        0,
        n,
        n
    )


    TREES <- vector(
        "list",
        n_trees
    )


    # ==================================================================
    # TREE ENSEMBLE
    # ==================================================================

    for(b in seq_len(n_trees)) {

        # --------------------------------------------------------------
        # Feature sampling
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
        # PCA
        # --------------------------------------------------------------

        pca <- prcomp(
            DATA_s,
            center = TRUE,
            scale. = scale
        )


        eigenvalues <- pca$sdev^2

        prob <- eigenvalues /
            sum(eigenvalues)


        selected_pc <- sample(
            seq_along(prob),
            size = 1,
            prob = prob
        )


        scores <- drop(
            pca$x[
                ,
                selected_pc
            ]
        )


        # ==============================================================
        # FEATURE CONTRIBUTION OF SELECTED PC
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
            is.finite(contribution_sum) &&
            contribution_sum > 0
        ) {

            feature_contribution <-
                feature_contribution /
                contribution_sum

        } else {

            feature_contribution[] <- 0
        }


        # Original feature IDs.
        #
        # Duplicate names are intentional when replace = TRUE.

        names(feature_contribution) <-
            as.character(ids)


        # --------------------------------------------------------------
        # Hierarchical clustering
        # --------------------------------------------------------------

        hc <- fastcluster::hclust(
            dist(scores),
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


        for(l in seq_len(levels)) {

            cl <- cutree(
                hc,
                k = l + 1
            )


            LEVEL_CLUSTERS[[l]] <-
                cl


            cluster_ids <- sort(
                unique(cl)
            )


            centroids <- numeric(
                length(cluster_ids)
            )


            names(centroids) <-
                as.character(
                    cluster_ids
                )


            for(cc in seq_along(
                cluster_ids
            )) {

                cid <- cluster_ids[cc]

                centroids[cc] <- mean(
                    scores[
                        cl == cid
                    ]
                )
            }


            LEVEL_CENTROIDS[[l]] <-
                centroids


            AFF <- AFF +
                .TAPIO_association(
                    cl
                )
        }


        # --------------------------------------------------------------
        # Frozen tree
        # --------------------------------------------------------------

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

            train_scores =
                scores,

            level_clusters =
                LEVEL_CLUSTERS,

            level_centroids =
                LEVEL_CENTROIDS
        )
    }


    # ==================================================================
    # FINAL AFFINITY
    # ==================================================================

    AFF <- AFF /
        (
            n_trees *
            levels
        )


    DIST <- 1 - AFF

    diag(DIST) <- 0


    final_hclust <- fastcluster::hclust(
        as.dist(DIST),
        method = method
    )


    final_clusters <- cutree(
        final_hclust,
        k = k
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

        feature_names =
            feature_names,

        method =
            method,

        scale =
            scale,

        replace =
            replace,

        pca_selection =
            "random_weighted"
    )


    class(model) <-
        "inductiveTAPIO"


    return(model)
}


# ======================================================================
# PROJECT NEW DATA THROUGH ONE TREE
# ======================================================================

.TAPIO_project_tree <- function(
    tree,
    newdata
) {

    X <- as.matrix(newdata)


    Xs <- X[
        ,
        tree$feature_ids,
        drop = FALSE
    ]


    Xs <- sweep(
        Xs,
        2,
        tree$pca_center,
        "-"
    )


    if(
        !is.null(tree$pca_scale) &&
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


    drop(
        Xs %*%
        tree$pca_rotation
    )
}


# ======================================================================
# ONE TREE: NEW -> TRAIN AFFINITY
# ======================================================================

.TAPIO_tree_affinity <- function(
    tree,
    new_scores
) {

    n_new <- length(
        new_scores
    )


    n_train <- length(
        tree$train_scores
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


        centroids <-
            tree$level_centroids[[l]]


        cluster_ids <-
            as.integer(
                names(centroids)
            )


        for(i in seq_len(n_new)) {

            d <- (
                new_scores[i] -
                centroids
            )^2


            new_cl <-
                cluster_ids[
                    which.min(d)
                ]


            A[i, ] <- A[i, ] +
                as.numeric(
                    train_cl ==
                    new_cl
                )
        }
    }


    return(A)
}


# ======================================================================
# ENSEMBLE NEW -> TRAIN AFFINITY
# ======================================================================

.TAPIO_predict_affinity <- function(
    object,
    newdata
) {

    X <- as.matrix(newdata)


    A <- matrix(
        0,
        nrow(X),
        length(
            object$train_clusters
        )
    )


    for(b in seq_len(
        object$n_trees
    )) {

        z <- .TAPIO_project_tree(
            object$trees[[b]],
            X
        )


        A <- A +
            .TAPIO_tree_affinity(
                object$trees[[b]],
                z
            )
    }


    A <- A /
        (
            object$n_trees *
            object$levels
        )


    return(A)
}


# ======================================================================
# S3 PREDICT METHOD
# ======================================================================

predict.inductiveTAPIO <- function(
    object,
    newdata,
    ...
) {

    X <- as.matrix(newdata)


    if(anyNA(X)) {
        stop(
            "Missing values are currently not supported."
        )
    }


    if(
        ncol(X) !=
        object$n_features_total
    ) {

        stop(
            "newdata has a different number of features."
        )
    }


    if(
        !is.null(colnames(X)) &&
        !identical(
            colnames(X),
            object$feature_names
        )
    ) {

        stop(
            "Feature names/order do not match training data."
        )
    }


    Anew <- .TAPIO_predict_affinity(
        object,
        X
    )


    cluster_scores <- matrix(
        NA_real_,
        nrow(X),
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


    predicted <- max.col(
        cluster_scores,
        ties.method = "first"
    )


    margin <- apply(

        cluster_scores,

        1,

        function(x) {

            sx <- sort(
                x,
                decreasing = TRUE
            )

            if(length(sx) < 2) {
                return(NA_real_)
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

            confidence =
                margin,

            affinity =
                Anew
        )
    )
}