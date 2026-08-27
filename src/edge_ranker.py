""" Edge-ranking model for the cascading_failure_sequence PowerGraph sub-task.

Architecture: a GINEConv backbone produces node embeddings; each edge's
score is computed from the concatenation of its two endpoint embeddings
plus its own edge features, via a small MLP head. The model outputs one
score per edge; higher score = predicted to fail earlier in the cascade.

"""
import torch
from torch_geometric.nn import GINEConv


class EdgeRanker(torch.nn.Module):
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

        # edge score head: takes [source node embedding, target node
        # embedding, edge features] -> scalar score
        self.edge_head = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels + edge_in, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        """ Returns a score per edge, shape (n_edges,). """
        h = self.conv1(x, edge_index, edge_attr).relu()
        h = self.conv2(h, edge_index, edge_attr).relu()

        src, dst = edge_index[0], edge_index[1]
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        scores = self.edge_head(edge_repr).squeeze(-1)
        return scores
