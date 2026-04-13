import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances


def build_temporal_graph(df, k_similarity=5):
    df = df.copy()

    # -------------------------------------------------
    # STEP 1 — SAFE wide format (longitudinal matrix)
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
    X = wide[feature_cols].values

    # -------------------------------------------------
    # STEP 2 — stable node ids
    # -------------------------------------------------
    wide["node_id"] = np.arange(len(wide))

    # -------------------------------------------------
    # STEP 3 — patient-level labels
    # -------------------------------------------------
    patient_labels = (
        df[["subject", "cluster"]]
        .drop_duplicates()
        .sort_values("subject")
    )

    # -------------------------------------------------
    # STEP 4 — build graph
    # -------------------------------------------------
    G = nx.Graph()

    # IMPORTANT FIX: attach features here
    for i, row in wide.iterrows():
        G.add_node(
            row["node_id"],
            subject=int(row["subject"]),
            time=float(row["time"]),
            features=np.asarray(X[i], dtype=np.float32)  # 🔥 FIXED
        )

    # -------------------------------------------------
    # STEP 5 — temporal edges (within subject)
    # -------------------------------------------------
    node_index = {
        (r.subject, r.time): r.node_id
        for _, r in wide.iterrows()
    }

    for subject in wide["subject"].unique():
        sub = wide[wide["subject"] == subject].sort_values("time")

        times = sub["time"].values

        for t1, t2 in zip(times[:-1], times[1:]):
            i = node_index[(subject, t1)]
            j = node_index[(subject, t2)]
            G.add_edge(i, j, edge_type="temporal")

    # -------------------------------------------------
    # STEP 6 — similarity edges (kNN in feature space)
    # -------------------------------------------------
    dist = pairwise_distances(X)

    for i in range(len(X)):
        neighbors = np.argsort(dist[i])[1:k_similarity + 1]
        for j in neighbors:
            G.add_edge(i, j, edge_type="similarity")

    return G, patient_labels