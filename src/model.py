from __future__ import annotations

import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer("adjacency", adjacency)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("ij,btjf->btif", self.adjacency, x)
        return self.linear(support)


class GraphGRUForecaster(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.graph_conv = GraphConvolution(input_dim, hidden_dim, adjacency)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_steps, nodes, features]
        x = torch.relu(self.graph_conv(x))
        batch_size, input_steps, num_nodes, hidden_dim = x.shape

        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, input_steps, hidden_dim)
        _, hidden = self.gru(x)
        hidden = hidden[-1].reshape(batch_size, num_nodes, self.hidden_dim)

        pred = self.readout(hidden)
        return pred.permute(0, 2, 1).unsqueeze(-1)
