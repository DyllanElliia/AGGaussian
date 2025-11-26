import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg

import numpy as np
"""
We did not use them in this work.
"""


class GCNEdgeNet(nn.Module):

  def __init__(self, gnn_in, gnn_hidden, gnn_out, mlp_hidden, mlp_out):
    super(GCNEdgeNet, self).__init__()

    self.conv = nn.ModuleList([
        tg.nn.GCNConv(gnn_in, gnn_hidden),
        tg.nn.GCNConv(gnn_hidden, gnn_out),
    ])
    # self.conv = nn.ModuleList([
    #     tg.nn.GATConv(gnn_in, gnn_hidden),
    #     tg.nn.GATConv(gnn_hidden, gnn_out),
    # ])

    self.mlp = nn.Sequential(nn.Linear(gnn_out, mlp_hidden), nn.ReLU(),
                             nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
                             nn.Linear(mlp_hidden, mlp_out), nn.Sigmoid())

  def cal_node_feature(self, x, edge_index):
    for conv in self.conv:
      x = conv(x, edge_index)
      x = F.relu(x)
    return x

  def predict_edge_prob(self, x, edge_index):
    """
    Args:
      x: (num_edges, hidden_channels)
      edge_index: (2, num_edges)
    """
    row, col = edge_index
    # edge_attr = torch.cat([x[row], x[col]], dim=1)
    edge_attr = x[row] - x[col]
    return self.mlp(edge_attr)

  def forward(self, x, edge_index):
    """
    Args:
      x: (num_nodes, in_channels)
      edge_index: (num_edges, 2)
    """
    edge_index = edge_index.t().contiguous()  # (2, num_edges)
    # change the edges to undirected
    edge_index_undirect = torch.cat([edge_index, edge_index[[1, 0]]],
                                    dim=1)  # (2, 2 * num_edges)
    f = self.cal_node_feature(x, edge_index_undirect)
    return self.predict_edge_prob(f, edge_index)


class GATEdgeNet(nn.Module):

  def __init__(self, gnn_in, gnn_hidden, gnn_out, mlp_hidden, mlp_out,
               edge_dist_dim):
    super(GATEdgeNet, self).__init__()

    self.conv = nn.ModuleList([
        tg.nn.GATConv(gnn_in, gnn_hidden),
        tg.nn.GATConv(gnn_hidden, gnn_out),
    ])

    self.mlp = nn.Sequential(
        nn.Linear((gnn_in + gnn_out) * 2 + edge_dist_dim, mlp_hidden),
        nn.ReLU(), nn.Linear(mlp_hidden, mlp_out), nn.Sigmoid())

  def cal_node_feature(self, x, edge_index):
    for conv in self.conv:
      x = conv(x, edge_index)
      x = F.relu(x)
    return x

  def predict_edge_prob(self, x, edge_index, edge_dist):
    """
    Args:
      x: (num_edges, hidden_channels)
      edge_index: (2, num_edges)
      edge_dist: (num_edges, edge_dist_dim)
    """
    row, col = edge_index
    # edge_attr = torch.cat([x[row], x[col]], dim=1)
    # edge_attr = x[row] - x[col]
    edge_attr = torch.cat([x[row], x[col], edge_dist], dim=1)
    # edge_attr = torch.cat(
    #     [x[row], x[col],
    #      torch.norm(x[row] - x[col], dim=1, keepdim=True)],
    #     dim=1)
    return self.mlp(edge_attr)

  def forward(self, x, edge_index, edge_dist):
    """
    Args:
      x: (num_nodes, in_channels)
      edge_index: (num_edges, 2)
    """
    edge_index = edge_index.t().contiguous()  # (2, num_edges)
    # change the edges to undirected
    edge_index_undirect = torch.cat([edge_index, edge_index[[1, 0]]],
                                    dim=1)  # (2, 2 * num_edges)
    f = self.cal_node_feature(x, edge_index_undirect)
    return self.predict_edge_prob(torch.cat([x, f], dim=1), edge_index,
                                  edge_dist)


class MLPNet(nn.Module):

  def __init__(self, gnn_in, gnn_hidden, gnn_out, mlp_hidden, mlp_out,
               edge_dist_dim):
    super(MLPNet, self).__init__()

    self.mlp = nn.Sequential(nn.Linear(gnn_in * 2 + edge_dist_dim, mlp_hidden),
                             nn.ReLU(), nn.Linear(mlp_hidden, mlp_out),
                             nn.Sigmoid())

  def predict_edge_prob(self, x, edge_index, edge_dist):
    """
    Args:
      x: (num_edges, hidden_channels)
      edge_index: (2, num_edges)
      edge_dist: (num_edges, edge_dist_dim)
    """
    row, col = edge_index
    # edge_attr = torch.cat([x[row], x[col]], dim=1)
    # edge_attr = x[row] - x[col]
    edge_attr = torch.cat([x[row], x[col], edge_dist], dim=1)
    # edge_attr = torch.cat(
    #     [x[row], x[col],
    #      torch.norm(x[row] - x[col], dim=1, keepdim=True)],
    #     dim=1)
    return self.mlp(edge_attr)

  def forward(self, x, edge_index, edge_dist):
    """
    Args:
      x: (num_nodes, in_channels)
      edge_index: (num_edges, 2)
    """
    edge_index = edge_index.t().contiguous()  # (2, num_edges)

    return self.predict_edge_prob(x, edge_index, edge_dist)
