""" Graph-topological feature computation and PyG data conversion for PowerGraph.

"""
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def structural_features(edge_index, num_nodes):
    """ Compute degree, betweenness centrality, and clustering coefficient
    per node, given a 0-indexed edge_index array of shape (n_edges, 2).
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edge_index.astype(int).tolist())

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)

    feats = np.zeros((num_nodes, 3), dtype=np.float32)
    for i in range(num_nodes):
        feats[i, 0] = degree.get(i, 0)
        feats[i, 1] = betweenness.get(i, 0.0)
        feats[i, 2] = clustering.get(i, 0.0)
    return feats


def to_pyg_data(sample, include_structural=True):
    """ Convert one PowerGraph data record (as returned by aigrids.load)
    into a PyTorch Geometric Data object.
    """
    x_node = sample["x_node"]
    edge_index_raw = sample["edge_index"] - 1  # convert to 0-indexed

    if include_structural:
        struct_feats = structural_features(edge_index_raw, x_node.shape[0])
        x = np.concatenate([x_node, struct_feats], axis=1)
    else:
        x = x_node

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(edge_index_raw, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(sample["x_edge"], dtype=torch.float)
    y = torch.tensor([sample["labels"]], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def to_pyg_dataset(data_list, include_structural=True):
    """ Convert a list of PowerGraph data records into a list of PyG Data objects. """
    return [to_pyg_data(s, include_structural=include_structural) for s in data_list]
