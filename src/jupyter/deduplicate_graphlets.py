import pickle as pkl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import pandas as pd 

def get_unique_graphlets(graphlets):
    unique_graphlets = []
    seen = set()
    for g in graphlets:
        g_tuple = tuple(sorted(g))
        if g_tuple not in seen:
            seen.add(g_tuple)
            unique_graphlets.append(g)
    return np.array(unique_graphlets)


for i in range(1, 50):
    graphlets = np.loadtxt(f"edgelists/tx_{i}.graphlets", dtype=int)
    graphlets = get_unique_graphlets(graphlets)
    np.savetxt(f"graphlets/tx_{i}_dedup_6.graphlets", graphlets, fmt='%d')
