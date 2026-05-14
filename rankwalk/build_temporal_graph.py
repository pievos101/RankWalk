import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def _robust_distance_matrix(X, n_perturb=5, noise_level=0.01):
    """
    Build a robust distance matrix via ensemble perturbations.
    Reduces sensitivity to single-feature corruption.
    """

    n = X.shape[0]
    dist_accum = np.zeros((n, n), dtype=np.float32)

    for _ in range(n_perturb):

        X_pert = X.copy()

        # small gaussian noise (prevents brittle kNN boundaries)
        X_pert += np.random.normal(0, noise_level, X_pert.shape)

        # robust scaling per feature
        X_pert = StandardScaler().fit_transform(X_pert)

        dist_accum += pairwise_distances(X_pert)

    return dist_accum / n_perturb


def build_temporal_graph(df, k_similarity=10):
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

    feature_cols = [c for c in wide.columns if c not in ["subject", "time"]]
    X = wide[feature_cols].values.astype(np.float32)

    # -------------------------------------------------
    # STEP 2 — robust feature preprocessing
    # -------------------------------------------------
    X = np.nan_to_num(X)

    # clip extreme values (IMPORTANT for robustness)
    q_low = np.quantile(X, 0.01, axis=0)
    q_high = np.quantile(X, 0.99, axis=0)
    X = np.clip(X, q_low, q_high)

    # standardize
    X = StandardScaler().fit_transform(X)

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
        sub = wide[wide["subject"] == subject].sort_values("time")

        for t1, t2 in zip(sub["time"].values[:-1], sub["time"].values[1:]):
            i = node_index[(subject, t1)]
            j = node_index[(subject, t2)]
            G.add_edge(i, j, edge_type="temporal")

    # -------------------------------------------------
    # STEP 7 — ROBUST similarity edges (KEY CHANGE)
    # -------------------------------------------------
    dist = _robust_distance_matrix(X)

    for t in wide["time"].unique():

        slice_idx = wide.index[wide["time"] == t].to_numpy()

        if len(slice_idx) <= k_similarity:
            continue

        dist_slice = dist[np.ix_(slice_idx, slice_idx)]

        for i_local, i_global in enumerate(slice_idx):

            # stable kNN via robust distances
            neighbors = np.argsort(dist_slice[i_local])[1:k_similarity + 1]

            for j_local in neighbors:

                j_global = slice_idx[j_local]

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type="similarity",
                    time=float(t)
                )

    return G, patient_labels