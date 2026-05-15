import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


def _robust_distance_matrix(
    X,
    n_subspaces=5,
    subspace_ratio=0.7
):
    """
    SAME robustness strategy as the R version:

    - random feature subspaces
    - Euclidean distances per subspace
    - rank aggregation across subspaces
    """

    n, p = X.shape

    subspace_size = max(
        2,
        int(np.floor(subspace_ratio * p))
    )

    dist_accum = np.zeros((n, n), dtype=np.float32)

    for _ in range(n_subspaces):

        # random feature subset
        feat_idx = np.random.choice(
            p,
            size=subspace_size,
            replace=False
        )

        X_sub = X[:, feat_idx]

        # pairwise Euclidean distance
        d = squareform(
            pdist(X_sub, metric="euclidean")
        )

        # SAME AS R: rank(d)
        ranked = (
            d.ravel()
            .argsort()
            .argsort()
            .reshape(d.shape)
        )

        dist_accum += ranked

    return dist_accum / n_subspaces


def build_temporal_graph(
    df,
    k_similarity=10,
    n_subspaces=5,
    subspace_ratio=0.7
):
    df = df.copy()

    # -------------------------------------------------
    # STEP 1 — wide format
    # -------------------------------------------------
    wide = (
        df.groupby(["subject", "time", "outcome"])["y"]
        .mean()
        .unstack("outcome")
        .reset_index()
        .sort_values(["subject", "time"])
        .reset_index(drop=True)
    )

    feature_cols = [
        c for c in wide.columns
        if c not in ["subject", "time"]
    ]

    X = wide[feature_cols].values.astype(np.float32)

    # -------------------------------------------------
    # STEP 2 — SAME normalization as R
    # -------------------------------------------------
    X = np.nan_to_num(X)

    # equivalent to R: scale(X_raw)
    X = StandardScaler().fit_transform(X)

    # SAME clipping as R
    clip_val = 4
    X[X > clip_val] = clip_val
    X[X < -clip_val] = -clip_val

    # -------------------------------------------------
    # STEP 3 — node ids
    # -------------------------------------------------
    wide["node_id"] = np.arange(len(wide))

    # -------------------------------------------------
    # STEP 4 — labels
    # -------------------------------------------------
    patient_labels = (
        df[["subject", "cluster"]]
        .drop_duplicates()
        .sort_values("subject")
    )

    # -------------------------------------------------
    # STEP 5 — graph
    # -------------------------------------------------
    G = nx.Graph()

    for i, row in wide.iterrows():

        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=X[i]
        )

    # -------------------------------------------------
    # STEP 6 — temporal edges
    # -------------------------------------------------
    node_index = {
        (r.subject, r.time): r.node_id
        for _, r in wide.iterrows()
    }

    for subject in wide["subject"].unique():

        sub = (
            wide[wide["subject"] == subject]
            .sort_values("time")
        )

        for t1, t2 in zip(
            sub["time"].values[:-1],
            sub["time"].values[1:]
        ):

            i = node_index[(subject, t1)]
            j = node_index[(subject, t2)]

            G.add_edge(
                i,
                j,
                edge_type="temporal"
            )

    # -------------------------------------------------
    # STEP 7 — SAME slice-based similarity as R
    # -------------------------------------------------
    for t in sorted(wide["time"].unique()):

        slice_idx = wide.index[
            wide["time"] == t
        ].to_numpy()

        if len(slice_idx) <= k_similarity:
            continue

        # SAME AS R:
        # X_slice <- X[idx, , drop = FALSE]
        X_slice = X[slice_idx]

        # SAME random-subspace ensemble
        dist_slice = _robust_distance_matrix(
            X_slice,
            n_subspaces=n_subspaces,
            subspace_ratio=subspace_ratio
        )

        for i_local, i_global in enumerate(slice_idx):

            # SAME AS R:
            # neighbors <- order(...)[2:(k+1)]
            neighbors = np.argsort(
                dist_slice[i_local]
            )[1:k_similarity + 1]

            for j_local in neighbors:

                j_global = slice_idx[j_local]

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type="similarity",
                    time=float(t)
                )

    return G, patient_labels