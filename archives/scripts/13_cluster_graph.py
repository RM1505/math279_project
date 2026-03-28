import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from clsz.cluster import cluster_networkx
import pandas as pd

A_spar = np.load("data/processed/A_sparse.npy")   # (N,N), signed, sparse-ish
W = np.abs(A_spar).astype(np.float64)
np.fill_diagonal(W, 0.0)

G = nx.from_numpy_array(W, create_using=nx.DiGraph)

labels = cluster_networkx(G, 10)
print(labels)

df = pd.read_csv("data/index.csv")
df["cluster"] = labels
df = df[["ticker", "cluster"]]
df.to_csv("data/processed/clusters.csv", index=True)
