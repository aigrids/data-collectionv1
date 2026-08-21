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


def _convert_label(label, task_type):
    """ Convert a raw PowerGraph label into the tensor format expected
    by the loss function for a given task type.

    - binary: raw label is already a float (0.0 / 1.0) -> shape (1,)
    - multiclass: raw label is a one-hot list, e.g. [0,0,0,1] -> class
      index as a LongTensor, shape (1,), for use with cross_entropy
    - regression: raw label is already a float -> shape (1,)
    """
    if task_type == "multiclass":
        class_idx = int(np.argmax(label))
        return torch.tensor([class_idx], dtype=torch.long)
    else:
        return torch.tensor([label], dtype=torch.float)


def to_pyg_data(sample, task_type="binary", include_structural=True):
    """ Convert one PowerGraph data record (as returned by aigrids.load)
    into a PyTorch Geometric Data object.

    task_type: one of 'binary', 'multiclass', 'regression'. Determines
    how sample["labels"] is converted (see _convert_label).
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
    y = _convert_label(sample["labels"], task_type)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def to_pyg_dataset(data_list, task_type="binary", include_structural=True):
    """ Convert a list of PowerGraph data records into a list of PyG Data objects. """
    return [to_pyg_data(s, task_type=task_type, include_structural=include_structural)
            for s in data_list]


# maps each PowerGraph subtask name to its task type and number of output classes
SUBTASK_CONFIG = {
    "cascading_failure_binary": {"task_type": "binary", "out_channels": 1},
    "cascading_failure_multiclass": {"task_type": "multiclass", "out_channels": 4},
    "demand_not_served_regression": {"task_type": "regression", "out_channels": 1},
    # cascading_failure_sequence intentionally omitted: requires a
    # fundamentally different sequence-prediction architecture, not
    # a fixed-size classification/regression head.
}
