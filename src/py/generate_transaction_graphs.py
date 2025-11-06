import pandas as pd
import networkx as nx
import pickle
import os

path = "../../Elliptic_plusplus"

df_tx_edges = pd.read_csv(f"{path}/txs_edgelist.csv")
df_tx_feats = pd.read_csv(f"{path}/txs_features.csv")
df_tx_classes = pd.read_csv(f"{path}/txs_classes.csv")

# Combine node features and class
df_combined_feats = df_tx_feats.merge(df_tx_classes, on="txId", how="inner")

# Copy column so we can merge
df_tx_edges["txId"] = df_tx_edges["txId1"]

# Merge on txId1 to add Time Step -- Shouldn't matter on which one we do, however good to keep in mind
df_edges = df_tx_edges.merge(df_tx_feats[["Time step", "txId"]], on="txId", how="inner").drop(columns=["txId"]).drop_duplicates()

# Prepare node features
df_feats = df_combined_feats.set_index("txId")

for dt in range(1, 50):
    print("Timestep:", dt)
    df_feats_dt = df_feats[df_feats["Time step"] == dt]
    
    df = df_edges[df_edges["Time step"] == dt]
    G = nx.from_pandas_edgelist(df, "txId1", "txId2", create_using=nx.DiGraph())
    for i, node in enumerate(G.nodes()):
        if i % 1000 == 0:
            print(f"Node {i+1}/{G.number_of_nodes()}")
            
        try:
            row = df_feats_dt.loc[node]
        except KeyError:
            raise ValueError(f"Node {node} not found in feature df")
        
        G.nodes[node].update(row.to_dict())
        

    with open(f"../../graphs/tx_graph/tx_transaction_graph_timestamp_{dt}.pkl", "wb") as f:
        pickle.dump(G, f)

