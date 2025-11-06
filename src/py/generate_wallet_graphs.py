# This notebook constructs graphs for each time step using networkx.

import pandas as pd
import networkx as nx
import pickle
import os 

path = "../../Elliptic_plusplus"
output_path = "../../graphs"

df_addr_tx = pd.read_csv(f'{path}/AddrTx_edgelist.csv')
df_tx_addr = pd.read_csv(f'{path}/TxAddr_edgelist.csv')
df_tx_feats = pd.read_csv(f'{path}/txs_features.csv')
df_addr_feats = pd.read_csv(f'{path}/wallets_features_classes_combined.csv')

# Compare node and edge features to see if we need both

df_edges = df_addr_tx.merge(df_tx_addr, on="txId", how="inner")
df_edges = df_edges.merge(df_tx_feats[["txId", "Time step"]], on="txId", how="inner")

df_edges = df_edges.drop_duplicates(keep="first")
df_addr_feats = df_addr_feats.drop_duplicates(keep="first")
df_tx_feats = df_tx_feats.drop_duplicates(keep="first")

# Add index to dataframes for faster lookup
df_addr_feats_indexed = df_addr_feats.set_index('address')
df_tx_feats_indexed = df_tx_feats.set_index('txId')

for dt in range(1, 50):
    if os.path.exists(f"{output_path}/address_transaction_graph_timestep_{dt}.pkl"):
        print(f"Graph for time step {dt} already exists. Skipping...")
        continue
    
    print(f"Constructing graph for time step {dt}...")
    df_dt = df_edges[df_edges['Time step'] == dt]   # Get relevant data for graph

    # Create nodes and edges
    G_dt = nx.from_pandas_edgelist(df_dt, 'input_address', 'output_address', edge_attr="txId", create_using=nx.MultiDiGraph())

    for c, node in enumerate(G_dt.nodes()):
        if c % 1000 == 0:
            print(f"Processing node {c}/{G_dt.number_of_nodes()}")
        try:
            row = df_addr_feats_indexed.loc[node]
        except KeyError:
            raise ValueError(f"Node {node} not found in address features dataframe.")
            
        feats, label = row.drop(columns=["class"]), row["class"]
        G_dt.nodes[node].update(feats.to_dict())
        G_dt.nodes[node]['label'] = label
            

    for c, (u, v, k, data) in enumerate(G_dt.edges(keys=True, data=True)):
        if c % 1000 == 0:
            print(f"Processing edge {c}/{G_dt.number_of_edges()}")
        
        try:
            row = df_tx_feats_indexed.loc[data['txId']]
        except KeyError:
            raise ValueError(f"txId {data['txId']} not found in transaction features dataframe.")
            
        G_dt.edges[u, v, k].update(row.to_dict())        


    with open(f"{output_path}/address_transaction_graph_timestep_{dt}.pkl", "wb") as f:
        pickle.dump(G_dt, f)


