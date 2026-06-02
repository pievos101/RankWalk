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

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler

TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1

def build_temporal_graph_grid(
    df,
    n_bins=10,
    k_similarity=10,
    n_subspaces=5,
    subspace_ratio=0.7,
    overlap=0.5   # NEW: window overlap (0.3–0.7 recommended)
):

    import numpy as np
    import networkx as nx
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # =====================================================
    # SLIDING TIME WINDOWS (OVERLAPPING VISITS)
    # =====================================================
    t_min = df["time"].min()
    t_max = df["time"].max()

    total_range = t_max - t_min
    window_size = total_range / n_bins
    step = window_size * (1 - overlap)

    rows = []

    visit = 0
    start = t_min

    while start <= t_max:

        end = start + window_size

        tmp = df[
            (df["time"] >= start) &
            (df["time"] < end)
        ].copy()

        if len(tmp) > 0:
            tmp["visit"] = visit
            tmp["window_start"] = start
            tmp["window_end"] = end
            rows.append(tmp)

        visit += 1
        start += step

    df = pd.concat(rows, ignore_index=True)

    # =====================================================
    # AGGREGATION (REAL OBS ONLY)
    # =====================================================
    df = (
        df.groupby(["subject", "visit", "outcome"], as_index=False)
        .agg(
            y=("y", "mean"),
            time=("time", "mean")
        )
    )

    # =====================================================
    # WIDE FORMAT
    # =====================================================
    wide = (
        df.pivot_table(
            index=["subject", "visit"],
            columns="outcome",
            values="y",
            aggfunc="mean"
        )
        .reset_index()
    )

    tmap = (
        df.groupby(["subject", "visit"], as_index=False)["time"]
        .mean()
    )

    wide = wide.merge(tmap, on=["subject", "visit"], how="left")

    # =====================================================
    # FEATURE MATRIX (NO TIME IN FEATURES)
    # =====================================================
    feature_cols = [
        c for c in wide.columns
        if c not in ["subject", "visit", "time"]
    ]

    X = wide[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X)

    X = StandardScaler().fit_transform(X)
    X = np.clip(X, -4, 4)

    # =====================================================
    # GRAPH INIT
    # =====================================================
    G = nx.Graph()

    wide = wide.sort_values(["subject", "visit"]).reset_index(drop=True)
    wide["node_id"] = np.arange(len(wide))

    node_index = {
        (r.subject, r.visit): r.node_id
        for r in wide.itertuples()
    }

    # =====================================================
    # NODES
    # =====================================================
    for i, row in wide.iterrows():

        G.add_node(
            int(row.node_id),
            subject=int(row.subject),
            visit=int(row.visit),
            time=float(row.time),
            features=X[i]
        )

    # =====================================================
    # TEMPORAL EDGES (SEQUENTIAL WITHIN SUBJECT)
    # =====================================================
    TEMPORAL_EDGE = 0
    SIMILARITY_EDGE = 1

    subjects = wide["subject"].unique()

    for s in subjects:

        sub = wide[wide.subject == s].sort_values("visit")

        for i in range(len(sub) - 1):

            u = node_index[(s, sub.iloc[i].visit)]
            v = node_index[(s, sub.iloc[i + 1].visit)]

            G.add_edge(
                u,
                v,
                edge_type=TEMPORAL_EDGE
            )

    # =====================================================
    # WITHIN-VISIT kNN (ROBUST SUBSPACE SIMILARITY)
    # =====================================================
    for v in sorted(wide["visit"].unique()):

        idx = wide.index[wide["visit"] == v].to_numpy()

        if len(idx) <= k_similarity:
            continue

        Xv = X[idx]

        dist = _robust_distance_matrix(
            Xv,
            n_subspaces=n_subspaces,
            subspace_ratio=subspace_ratio
        )

        for i_local, i_global in enumerate(idx):

            nn = np.argsort(dist[i_local])[1:k_similarity + 1]

            for j_local in nn:

                G.add_edge(
                    int(i_global),
                    int(idx[j_local]),
                    edge_type=SIMILARITY_EDGE,
                    visit=int(v)
                )

    print("nodes:", G.number_of_nodes())
    print("edges:", G.number_of_edges())

    return G, wide


import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler

TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1


# =====================================================
# ROBUST SUBSPACE CONSENSUS DISTANCE (KEY FIX)
# =====================================================
def robust_subspace_similarity(X, n_subspaces=5, subspace_ratio=0.7, tau=1.0):

    n, p = X.shape
    k = max(2, int(np.floor(subspace_ratio * p)))

    S_accum = np.zeros((n, n), dtype=np.float32)

    for _ in range(n_subspaces):

        feat_idx = np.random.choice(p, k, replace=False)
        Xs = X[:, feat_idx]

        # cosine similarity (more stable under noise than Euclidean)
        norm = np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-8
        Xs = Xs / norm

        S = Xs @ Xs.T
        S_accum += S

    S = S_accum / n_subspaces

    # temperature sharpening
    S = np.exp(S / tau)

    return S


# =====================================================
# MAIN GRAPH BUILDER (VERSION 2)
# =====================================================
def build_temporal_graph_grid2(
    df,
    n_bins=10,
    k_similarity=10,
    n_subspaces=5,
    subspace_ratio=0.7,
    tau=1.0
):

    df = df.copy()

    # =====================================================
    # STEP 1: QUANTILE BINNING (IRREGULAR TIME SAFE)
    # =====================================================
    df["visit"] = pd.qcut(
        df["time"].rank(method="first"),
        q=n_bins,
        labels=False
    ).astype(int)

    # =====================================================
    # STEP 2: AGGREGATE REAL OBSERVATIONS ONLY
    # =====================================================
    df = (
        df.groupby(["subject", "visit", "outcome"], as_index=False)
        .agg(y=("y", "mean"))
    )

    # =====================================================
    # STEP 3: WIDE FORMAT (NO FAKE NODES)
    # =====================================================
    wide = (
        df.pivot_table(
            index=["subject", "visit"],
            columns="outcome",
            values="y",
            aggfunc="mean"
        )
        .reset_index()
    )

    # =====================================================
    # STEP 4: FEATURES (NO TIME IN FEATURES)
    # =====================================================
    feature_cols = [
        c for c in wide.columns
        if c not in ["subject", "visit"]
    ]

    X = wide[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X)

    X = StandardScaler().fit_transform(X)
    X = np.clip(X, -4, 4)

    # =====================================================
    # STEP 5: GRAPH INIT
    # =====================================================
    G = nx.Graph()

    wide = wide.sort_values(["subject", "visit"]).reset_index(drop=True)
    wide["node_id"] = np.arange(len(wide))

    node_index = {
        (r.subject, r.visit): r.node_id
        for r in wide.itertuples()
    }

    # =====================================================
    # ADD NODES
    # =====================================================
    for i, row in wide.iterrows():

        G.add_node(
            int(row.node_id),
            subject=int(row.subject),
            visit=int(row.visit),
            features=X[i]
        )

    # =====================================================
    # STEP 6: TEMPORAL EDGES (STRUCTURE ONLY)
    # =====================================================
    subjects = wide["subject"].unique()

    for s in subjects:

        sub = wide[wide.subject == s].sort_values("visit")

        for i in range(len(sub) - 1):

            u = node_index[(s, sub.iloc[i].visit)]
            v = node_index[(s, sub.iloc[i + 1].visit)]

            G.add_edge(
                u,
                v,
                edge_type=TEMPORAL_EDGE
            )

    # =====================================================
    # STEP 7: ROBUST SIMILARITY EDGES (SUBSPACE CONSENSUS)
    # =====================================================
    for v in sorted(wide["visit"].unique()):

        idx = wide.index[wide["visit"] == v].to_numpy()

        if len(idx) <= k_similarity:
            continue

        Xv = X[idx]

        # 🔥 KEY FIX: robust similarity
        S = robust_subspace_similarity(
            Xv,
            n_subspaces=n_subspaces,
            subspace_ratio=subspace_ratio,
            tau=tau
        )

        for i_local, i_global in enumerate(idx):

            nn = np.argsort(S[i_local])[::-1][1:k_similarity + 1]

            for j_local in nn:

                G.add_edge(
                    int(i_global),
                    int(idx[j_local]),
                    edge_type=SIMILARITY_EDGE,
                    weight=float(S[i_local, j_local]),
                    visit=int(v)
                )

    print("nodes:", G.number_of_nodes())

    return G, wide



#### OTHERS!

import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1


# =========================================================
# STABLE DISTANCE (kept, but used only for ranking)
# =========================================================
def stable_rank_distance(X):
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    return d


# =========================================================
# MUTUAL kNN MASK
# =========================================================
def mutual_knn_mask(D, k):
    n = D.shape[0]

    nn = np.argsort(D, axis=1)[:, 1:k+1]

    mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in nn[i]:
            mask[i, j] = True

    # mutual constraint
    mutual = mask & mask.T
    return mutual, nn


# =========================================================
# MAIN GRAPH BUILDER (ROBUST kNN VERSION)
# =========================================================
def build_temporal_graph_robust_knn(
    df,
    k_similarity=10
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
    X = np.clip(X, -clip_val, clip_val)

    # -----------------------------------------
    # TIME FEATURE (kept)
    # -----------------------------------------
    t = wide["time"].values.astype(np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)

    X = np.concatenate([X, t.reshape(-1, 1)], axis=1)

    # -----------------------------------------
    # NODE SETUP
    # -----------------------------------------
    wide["node_id"] = np.arange(len(wide))

    G = nx.Graph()

    for i, row in wide.iterrows():
        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=X[i]
        )

    # -----------------------------------------
    # TEMPORAL EDGES (UNCHANGED)
    # -----------------------------------------
    node_index = {
        (r.subject, r.time): r.node_id
        for _, r in wide.iterrows()
    }

    for subject in wide["subject"].unique():

        sub = wide[wide["subject"] == subject].sort_values("time")

        for t1, t2 in zip(sub["time"].values[:-1], sub["time"].values[1:]):

            i = node_index[(subject, t1)]
            j = node_index[(subject, t2)]

            G.add_edge(
                i, j,
                edge_type=TEMPORAL_EDGE,
                weight=1.0
            )

    # -----------------------------------------
    # ROBUST kNN PER TIME SLICE
    # -----------------------------------------
    for tval in sorted(wide["time"].unique()):

        idx = wide.index[wide["time"] == tval].to_numpy()

        if len(idx) <= k_similarity:
            continue

        X_slice = X[idx]

        # stable geometry
        D = stable_rank_distance(X_slice)

        mutual_mask, nn = mutual_knn_mask(D, k_similarity)

        for i_local, i_global in enumerate(idx):

            for j_local in nn[i_local]:

                # enforce mutual constraint (IMPORTANT)
                if not mutual_mask[i_local, j_local]:
                    continue

                j_global = idx[j_local]

                # convert distance → weight (stable)
                w = np.exp(-D[i_local, j_local])

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type=SIMILARITY_EDGE,
                    weight=float(w)
                )

    return G, None

###
# ANOTHER ONE
#####

import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1


# =========================================================
# ROBUST SUBSPACE DISTANCE
# =========================================================
def _final_robust_distance_matrix(
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

        # -----------------------------------------
        # Random feature subspace
        # -----------------------------------------
        feat_idx = np.random.choice(
            p,
            size=subspace_size,
            replace=False
        )

        X_sub = X[:, feat_idx]

        # -----------------------------------------
        # Euclidean distances in subspace
        # -----------------------------------------
        d = squareform(
            pdist(X_sub, metric="euclidean")
        )

        # -----------------------------------------
        # Rank-transform distances
        # (VERY important for robustness)
        # -----------------------------------------
        ranked = (
            d.ravel()
            .argsort()
            .argsort()
            .reshape(d.shape)
            .astype(np.float32)
        )

        dist_accum += ranked

    # averaged robust distance
    return dist_accum / n_subspaces


# =========================================================
# MUTUAL kNN
# =========================================================
def _final_mutual_knn_mask(
    D,
    k
):

    n = D.shape[0]

    nn = np.argsort(
        D,
        axis=1
    )[:, 1:k + 1]

    mask = np.zeros(
        (n, n),
        dtype=bool
    )

    for i in range(n):

        for j in nn[i]:

            mask[i, j] = True

    # -----------------------------------------
    # Mutual agreement
    # -----------------------------------------
    mutual = mask & mask.T

    return mutual, nn


# =========================================================
# FINAL GRAPH BUILDER
# =========================================================
def build_temporal_graph_final(
    df,
    k_similarity=10,
    n_subspaces=5,
    subspace_ratio=0.7
):

    df = df.copy()

    # =====================================================
    # WIDE FORMAT
    # =====================================================
    wide = (
        df.groupby(
            ["subject", "time", "outcome"]
        )["y"]
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

    # =====================================================
    # STANDARDIZATION
    # =====================================================
    X = np.nan_to_num(X)

    X = StandardScaler().fit_transform(X)

    clip_val = 4

    X = np.clip(
        X,
        -clip_val,
        clip_val
    )

    # =====================================================
    # EXPLICIT TIME FEATURE
    # =====================================================
    time_values = (
        wide["time"]
        .values
        .astype(np.float32)
    )

    time_scaled = (
        (time_values - time_values.mean()) /
        (time_values.std() + 1e-8)
    ).reshape(-1, 1)

    X = np.concatenate(
        [X, time_scaled],
        axis=1
    )

    # =====================================================
    # NODE IDS
    # =====================================================
    wide["node_id"] = np.arange(len(wide))

    patient_labels = (
        df[["subject", "cluster"]]
        .drop_duplicates()
        .sort_values("subject")
    )

    # =====================================================
    # GRAPH INITIALIZATION
    # =====================================================
    G = nx.Graph()

    for i, row in wide.iterrows():

        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=X[i]
        )

    # =====================================================
    # TEMPORAL EDGES
    # =====================================================
    node_index = {
        (r.subject, r.time): r.node_id
        for _, r in wide.iterrows()
    }

    for subject in wide["subject"].unique():

        sub = (
            wide[
                wide["subject"] == subject
            ]
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
                edge_type=TEMPORAL_EDGE,
                weight=1.0
            )

    # =====================================================
    # ROBUST MUTUAL kNN GRAPH
    # =====================================================
    for t in sorted(wide["time"].unique()):

        slice_idx = (
            wide.index[
                wide["time"] == t
            ]
            .to_numpy()
        )

        if len(slice_idx) <= k_similarity:
            continue

        X_slice = X[slice_idx]

        # -----------------------------------------
        # Robust subspace distance
        # -----------------------------------------
        D = _final_robust_distance_matrix(
            X_slice,
            n_subspaces=n_subspaces,
            subspace_ratio=subspace_ratio
        )

        # -----------------------------------------
        # Mutual kNN
        # -----------------------------------------
        mutual_mask, nn = _final_mutual_knn_mask(
            D,
            k_similarity
        )

        # -----------------------------------------
        # Build similarity graph
        # -----------------------------------------
        for i_local, i_global in enumerate(slice_idx):

            for j_local in nn[i_local]:

                # IMPORTANT:
                # keep only mutual neighbors
                if not mutual_mask[i_local, j_local]:
                    continue

                j_global = slice_idx[j_local]

                # ---------------------------------
                # Stable similarity weight
                # ---------------------------------
                w = np.exp(
                    -D[i_local, j_local] /
                    (D.std() + 1e-8)
                )

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type=SIMILARITY_EDGE,
                    weight=float(w),
                    time=float(t)
                )

    return G, patient_labels

####

import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler

TEMPORAL_EDGE = 0
SIMILARITY_EDGE = 1


# =========================================================
# STABLE DISTANCE
# =========================================================
def stable_rank_distance(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)


# =========================================================
# MUTUAL kNN (FIXED + CLEAN)
# =========================================================
def mutual_knn(D, k):
    nn = np.argsort(D, axis=1)[:, 1:k+1]

    n = D.shape[0]
    mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        mask[i, nn[i]] = True

    mutual = mask & mask.T
    return nn, mutual


# =========================================================
# MAIN GRAPH BUILDER (NOW WITH KNN MODE SWITCH)
# =========================================================
def build_temporal_graph_aligned(
    df,
    k_similarity=10,
    k_align=5,
    knn_mode="mutual"   # <<<<<< NEW PARAMETER
):

    df = df.copy()

    # -----------------------------------------
    # WIDE FORMAT
    # -----------------------------------------
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

    # -----------------------------------------
    # STANDARDIZATION
    # -----------------------------------------
    X = np.nan_to_num(X)
    X = StandardScaler().fit_transform(X)
    X = np.clip(X, -4, 4)

    # time feature
    t = wide["time"].values.astype(np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)

    X = np.concatenate([X, t.reshape(-1, 1)], axis=1)

    wide["node_id"] = np.arange(len(wide))

    G = nx.Graph()

    for i, row in wide.iterrows():
        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=X[i]
        )

    # =====================================================
    # 1. TEMPORAL EDGES (subject continuity)
    # =====================================================
    node_index = {
        (r.subject, r.time): r.node_id
        for _, r in wide.iterrows()
    }

    for subject in wide["subject"].unique():

        sub = wide[wide["subject"] == subject].sort_values("time")

        for t1, t2 in zip(sub["time"].values[:-1], sub["time"].values[1:]):

            i = node_index[(subject, t1)]
            j = node_index[(subject, t2)]

            G.add_edge(i, j, edge_type=TEMPORAL_EDGE, weight=1.0)

    # =====================================================
    # 2. WITHIN-TIME kNN (SWITCHABLE)
    # =====================================================
    for tval in sorted(wide["time"].unique()):

        idx = wide.index[wide["time"] == tval].to_numpy()
        if len(idx) <= k_similarity:
            continue

        X_slice = X[idx]
        D = stable_rank_distance(X_slice)

        nn, mutual = mutual_knn(D, k_similarity)

        for i_local, i_global in enumerate(idx):

            for j_local in nn[i_local]:

                # -----------------------------------------
                # MODE SWITCH
                # -----------------------------------------
                if knn_mode == "mutual":
                    if not mutual[i_local, j_local]:
                        continue

                j_global = idx[j_local]

                w = np.exp(-D[i_local, j_local])

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type=SIMILARITY_EDGE,
                    weight=float(w)
                )

    # =====================================================
    # 3. CROSS-TIME ALIGNMENT (UNCHANGED)
    # =====================================================
    time_values = sorted(wide["time"].unique())

    for t1, t2 in zip(time_values[:-1], time_values[1:]):

        idx1 = wide.index[wide["time"] == t1].to_numpy()
        idx2 = wide.index[wide["time"] == t2].to_numpy()

        if len(idx1) == 0 or len(idx2) == 0:
            continue

        X1, X2 = X[idx1], X[idx2]

        sim = X1 @ X2.T
        sim /= (
            np.linalg.norm(X1, axis=1, keepdims=True) *
            np.linalg.norm(X2, axis=1, keepdims=True).T + 1e-8
        )

        nn_fwd = np.argsort(sim, axis=1)[:, -k_align:]
        nn_bwd = np.argsort(sim.T, axis=1)[:, -k_align:]

        for i_local, i_global in enumerate(idx1):

            for j_local in nn_fwd[i_local]:

                if i_local not in nn_bwd[j_local]:
                    continue

                j_global = idx2[j_local]

                G.add_edge(
                    int(i_global),
                    int(j_global),
                    edge_type=TEMPORAL_EDGE,
                    weight=float(sim[i_local, j_local])
                )

    return G, None