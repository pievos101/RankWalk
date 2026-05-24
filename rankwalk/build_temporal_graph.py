import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1


def _robust_distance_matrix(
    X,
    n_subspaces=5,
    subspace_ratio=0.7
):

    n, p = X.shape

    subspace_size = max(
        2,
        int(np.floor(subspace_ratio * p))
    )

    dist_accum = np.zeros((n, n), dtype=np.float32)

    for _ in range(n_subspaces):

        feat_idx = np.random.choice(
            p,
            size=subspace_size,
            replace=False
        )

        X_sub = X[:, feat_idx]

        d = squareform(
            pdist(X_sub, metric="euclidean")
        )

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

    # -----------------------------------------
    # Wide format
    # -----------------------------------------
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

    # -----------------------------------------
    # Standardization
    # -----------------------------------------
    X = np.nan_to_num(X)

    X = StandardScaler().fit_transform(X)

    clip_val = 4

    X[X > clip_val] = clip_val
    X[X < -clip_val] = -clip_val

    # -----------------------------------------
    # Add explicit time feature
    # -----------------------------------------
    time_values = wide["time"].values.astype(np.float32)

    time_scaled = (
        (time_values - time_values.mean()) /
        (time_values.std() + 1e-8)
    )

    time_scaled = time_scaled.reshape(-1, 1)

    X = np.concatenate(
        [X, time_scaled],
        axis=1
    )

    # -----------------------------------------
    # Node ids
    # -----------------------------------------
    wide["node_id"] = np.arange(len(wide))

    patient_labels = (
        df[["subject", "cluster"]]
        .drop_duplicates()
        .sort_values("subject")
    )

    # -----------------------------------------
    # Graph
    # -----------------------------------------
    G = nx.Graph()

    for i, row in wide.iterrows():

        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=X[i]
        )

    # -----------------------------------------
    # Temporal edges
    # -----------------------------------------
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
                edge_type=TEMPORAL_EDGE
            )

    # -----------------------------------------
    # Similarity edges
    # -----------------------------------------
    for t in sorted(wide["time"].unique()):

        slice_idx = wide.index[
            wide["time"] == t
        ].to_numpy()

        if len(slice_idx) <= k_similarity:
            continue

        X_slice = X[slice_idx]

        dist_slice = _robust_distance_matrix(
            X_slice,
            n_subspaces=n_subspaces,
            subspace_ratio=subspace_ratio
        )

        for i_local, i_global in enumerate(slice_idx):

            neighbors = np.argsort(
                dist_slice[i_local]
            )[1:k_similarity + 1]

            for j_local in neighbors:

                j_global = slice_idx[j_local]

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type=SIMILARITY_EDGE,
                    time=float(t)
                )

    return G, patient_labels