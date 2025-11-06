from torch.utils.data import Dataset
import numpy as np
import torch 
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import pickle
import os
from torch_geometric.data import Data


def nanstd(x, dim=0):
    mask = ~torch.isnan(x)
    mean = torch.sum(x * mask, dim=dim) / torch.sum(mask, dim=dim)
    diff_sq = (x - mean.unsqueeze(dim)) ** 2
    diff_sq[~mask] = 0
    var = torch.sum(diff_sq, dim=dim) / torch.sum(mask, dim=dim)
    return torch.sqrt(var)


class GraphletDataset(Dataset):
    def __init__(self, graphlets=None, G=None, labeled_only=False, data_list=None):
        if data_list is not None:
            self.data_list = data_list
            self._extract_labels_from_loaded_data()
            return
        
        self.graphlets = graphlets
        self.G = G

        # Cache node attribute names
        self._cache_attrs()
        print(self.attrs)
        # Precompute labels
        self.labels = torch.tensor([self._get_single_graphlet_label(g) for g in graphlets], dtype=torch.long)

        # Keep only labeled graphlets if requested
        if labeled_only:
            mask = self.labels != 2
            self.graphlets = [g for g, keep in zip(self.graphlets, mask) if keep]
            self.labels = self.labels[mask]

        # Precompute normalization stats once
        self._precompute_standardization_params()

        # Precompute all node features in a tensor
        self._precompute_node_features()
        
        # Precompute and store all Data objects
        self.data_list = [self._graphlet_to_data(g) for g in self.graphlets]
        for d, y in zip(self.data_list, self.labels):
            d.y = y

    def _cache_attrs(self):
        first_node = next(iter(self.G.nodes(data=True)))[1]
        self.attrs = [k for k in first_node.keys() if k != "class" and k != "label"]

    def _extract_labels_from_loaded_data(self):
        self.labels = []
        for data in self.data_list:
            self.labels.append(int(data.y))
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def _precompute_standardization_params(self):
        all_feats = torch.tensor(
            [[data.get(a, float('nan')) for a in self.attrs] for _, data in self.G.nodes(data=True)],
            dtype=torch.float32
        )
        self.mean = torch.nanmean(all_feats, dim=0)
        self.mean = torch.nan_to_num(self.mean, nan=0.0)
        self.std = nanstd(all_feats, dim=0)
        self.std = torch.nan_to_num(self.std, nan=1.0) + 1e-6

    def _precompute_node_features(self):
        self.node_feats = torch.tensor([
            [self.G.nodes[n].get(a, float('nan')) for a in self.attrs]
            for n in self.G.nodes()
        ], dtype=torch.float32)
        self.node_idx = {n: i for i, n in enumerate(self.G.nodes())}


    def _get_single_graphlet_label(self, graphlet):
        node_classes = [self.G.nodes[n].get("class", 3) - 1 for n in graphlet]
        if 0 in node_classes: 
            return 0
        return max(set(node_classes), key=node_classes.count)

    def get_label_distribution(self):
        unique, counts = torch.unique(self.labels, return_counts=True)
        distribution = {int(u): int(c) for u, c in zip(unique, counts)}
        return distribution

    def _graphlet_to_data(self, graphlet):
        # Extract node features directly (no NetworkX copy)
        idxs = torch.tensor([self.node_idx[n] for n in graphlet])
        x = self.node_feats[idxs]

        # Replace NaNs with per-column means
        col_mean = torch.nanmean(x, dim=0)
        col_mean = torch.nan_to_num(col_mean, nan=0.0)
        nan_mask = torch.isnan(x)
        if nan_mask.any():
            x[nan_mask] = col_mean.repeat(x.shape[0], 1)[nan_mask]

        # Standardize
        x = (x - self.mean) / self.std

        # Construct edge index efficiently
        edges = [(i, j) for i, ni in enumerate(graphlet)
                 for j, nj in enumerate(graphlet)
                 if (ni, nj) in self.G.edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return len(self.data_list)
    
    def save_dataset(self, filename):
        # data_cpu = []
        # for d in self.data_list:
        #     if isinstance(d, Data):
        #         data_cpu.append(d.to('cpu'))
        #     else:
        #         raise TypeError(f"Unexpected type {type(d)} in data_list")
        torch.save(self.data_list, filename)
    
def load_dataset(filename):
    # allowlist PyG classes needed for your dataset
    data_list = torch.load(filename, weights_only=False)
    return GraphletDataset(data_list=data_list)


# # TODO: Add support for multiple Gs and graphlet sets from different timestamps
# class GraphletDataset(Dataset):
#     def __init__(self, graphlets=None, G=None, labeled_only=False, data_list=None):
#         if data_list is not None:
#             self.data_list = data_list
#             return
        
#         self.graphlets = graphlets
#         self.G = G
#         self.labels = torch.tensor([self._get_single_graphlet_label(g) for g in graphlets], dtype=torch.long)

#         if labeled_only:
#             labeled_indices = [i for i, lbl in enumerate(self.labels) if lbl != 2]
#             self.graphlets = [self.graphlets[i] for i in labeled_indices]
#             self.labels = self.labels[labeled_indices]

#         self._cache_attrs()
#         self._precompute_standardization_params()

#         self.node_feats = torch.tensor([
#             [G.nodes[n].get(a, float('nan')) for a in self.attrs]
#             for n in G.nodes()
#         ], dtype=torch.float32)
#         self.node_idx = {n: i for i, n in enumerate(G.nodes())}

#         self.data_list = [self._graphlet_to_data(g, self.G) for g in self.graphlets]
#         for i, d in enumerate(self.data_list):
#             d.y = self.labels[i]

#     def _cache_attrs(self):
#         first_node = next(iter(self.G.nodes(data=True)))[1]
#         self.attrs = [k for k in first_node.keys() if k != "class"]

#     def _precompute_standardization_params(self):
#         all_feats = []
#         for _, data in self.G.nodes(data=True):
#             feat = [v for k, v in data.items() if k in self.attrs and k != "class"]
#             all_feats.append(feat)
#         all_feats = torch.tensor(all_feats, dtype=torch.float32)
#         self.mean = torch.nanmean(all_feats, dim=0)
#         self.mean = torch.nan_to_num(self.mean, nan=0.0)
#         self.std = nanstd(all_feats, dim=0) + 1e-6
#         self.std = torch.nan_to_num(self.std, nan=1.0)

#     def __len__(self):
#         return len(self.graphlets)

#     def get_label_distribution(self):
#         unique, counts = torch.unique(self.labels, return_counts=True)
#         distribution = {int(u): int(c) for u, c in zip(unique, counts)}
#         return distribution

#     def _get_single_graphlet_label(self, graphlet):
#         node_classes = [self.G.nodes[n].get("class", 3) - 1 for n in graphlet]
#         if 0 in node_classes:
#             return 0
#         elif all(l == 1 for l in node_classes):
#             return 1
#         return 2
 
#     def _graphlet_to_data(self, graphlet, G):
#         subgraph = G.subgraph(graphlet).copy()
#         data = from_networkx(subgraph, group_node_attrs=self.attrs)
        
#         column_mean = torch.nan_to_num(data.x.nanmean(dim=0), nan=0.0)
#         nan_mask = torch.isnan(data.x)
#         data.x[nan_mask] = column_mean.repeat(data.x.shape[0], 1)[nan_mask]

#         data.x = (data.x - self.mean) / torch.clamp(self.std, min=1e-6)

#         return data
 
#     def __getitem__(self, idx):
#         return self.data_list[idx]


#     def save_dataset(self, filename):
#         torch.save(self.data_list, filename)
    
# def load_dataset(filename):
#     data_list = torch.load(filename)
#     return GraphletDataset(data_list=data_list)
