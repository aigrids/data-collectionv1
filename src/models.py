""" Graph neural network model architectures for PowerGraph benchmarking.

"""
import torch
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool


class GCN(torch.nn.Module):
    """ Two-layer Graph Convolutional Network using node features only. """

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
    """ Two-layer Graph Isomorphism Network with edge features (GINEConv). """

    def __init__(self, node_in, edge_in, hidden_channels):
        super().__init__()
        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(node_in, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels), torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels))
        self.conv1 = GINEConv(mlp1, edge_dim=edge_in)
        self.conv2 = GINEConv(mlp2, edge_dim=edge_in)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr).relu()
        x = global_mean_pool(x, batch)
        return self.lin(x).squeeze(-1)
