""" Graph neural network model architectures for PowerGraph benchmarking.

Supports three prediction types:
- binary classification (single logit, sigmoid + BCE)
- multiclass classification (n_classes logits, softmax + cross-entropy)
- regression (single scalar output, MSE)

"""
import torch
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool


class GCN(torch.nn.Module):
    """ Two-layer Graph Convolutional Network using node features only.
    Binary classification only (single output).
    """

    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)


class GINE(torch.nn.Module):
    """ Two-layer Graph Isomorphism Network with edge features (GINEConv).

    Generalized to support binary classification, multiclass classification,
    and regression via the `out_channels` and `task_type` parameters.

    task_type: one of 'binary', 'multiclass', 'regression'.
    out_channels: 1 for binary/regression, n_classes for multiclass.
    """

    def __init__(self, node_in, edge_in, hidden_channels, out_channels=1, task_type="binary"):
        super().__init__()
        assert task_type in ("binary", "multiclass", "regression")
        self.task_type = task_type

        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(node_in, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(mlp1, edge_dim=edge_in)
        self.conv2 = GINEConv(mlp2, edge_dim=edge_in)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = global_mean_pool(x, batch)
        out = self.lin(x)
        if self.task_type in ("binary", "regression"):
            return out.squeeze(-1)
        return out  # multiclass: keep (batch, n_classes) shape for cross-entropy
