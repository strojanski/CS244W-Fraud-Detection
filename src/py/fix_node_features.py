import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle

# Some nodes had objecst as values in the attribute dictionary instead of simple values.
# This script fixes that issue.
# TODO check edge features

def fix_node(G, node_tuple):
    node, index = node_tuple
    node_id = node[0]
    attrs = node[1]
    
    fixed_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, dict):
            k, v = list(value.items())[0]
            fixed_attrs[key] = v
            
    G.remove_node(node_id)
    G.add_node(node_id, **fixed_attrs)

for dt in range(1, 50):
    graph_path = f"../../graphs/address_transaction_graph_timestep_{dt}.pkl"
    print(graph_path)

    with open(graph_path, "rb") as f:
        G = pickle.load(f)

    nodes_to_fix = []

    nodes = list(G.nodes(data=True))
    for i, n in enumerate(nodes):
        if type(n[1]["num_txs_as_sender"]) not in [float, int]:
            nodes_to_fix.append((n, i))

    for n in nodes_to_fix:
        fix_node(G, n)

    nodes = list(G.nodes(data=True))
    for i, n in enumerate(nodes):
        if type(n[1]["num_txs_as_sender"]) not in [float, int]:
            print(n)

    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
