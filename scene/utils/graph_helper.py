import os
import torch
from torch import nn
import torch_scatter
import numpy as np
from plyfile import PlyData, PlyElement

from einops import rearrange, repeat

from tqdm import tqdm

from .. import GaussianModel

from .graph_baseclass import GraphBaseClass


# Graph Helper with Bilateral Graph Laplacian
class GraphHelper_BilateralGraphLaplacian(GraphBaseClass):

  def __init__(self, logger, gaussian_model: GaussianModel):
    super(GraphHelper_BilateralGraphLaplacian,
          self).__init__(logger, gaussian_model)
    self.logger = logger

    self.tau = 0.05
    # self.mlp = None

  def init_model(self):
    self.logger.info("This class is not neural part.")

  def train(self):
    # self.mlp.train()
    # self.optimizer.zero_grad()
    # pass
    self.is_train = True

  def eval(self):
    # self.mlp.eval()
    # pass
    self.is_train = False

  def step(self, iteration):
    # self.optimizer.step()
    # self.optimizer.zero_grad()
    pass

  def save(self, path):
    super().save(path)
    # torch.save(self.mlp.state_dict(), path.replace(".pth", "_mlp.pth"))

  def load(self, path):
    super().load(path)
    # if self.mlp is None:
    #   self.init_model()
    # self.mlp.load_state_dict(
    #     torch.load(path.replace(".pth", "_mlp.pth"), map_location="cuda"))
    # self.mlp.eval()

  def cal_edge_weight(self, edges):

    self.logger.info("edges shape: {}".format(edges.shape))
    # init
    # points = self.model.scatter_mean_to_anchor(
    #     self.model.get_xyz).detach()  # (n, 3)
    features = self.model.segment_activation(
        self.model._feature).detach()  # (n, f)

    # mean_opacity = self.model.scatter_mean_to_anchor(
    #     self.model.get_opacity).detach()  # (n, 1)
    # color = self.model.scatter_mean_to_anchor(
    #     self.model._features_dc.reshape(-1, 3)).detach()  # (n, 1)
    # radius = self.model._anchor_radii.detach()  # (n, 1)
    # s0 = self.model._anchor_radii[edges[:, 0]]
    # s1 = self.model._anchor_radii[edges[:, 1]]

    def cal_weight(v0, v1, sigma, scale):
      return torch.exp(-torch.sum(
          (scale * (v0 - v1))**2, dim=1, keepdim=True) / (2 * sigma**2 + 1e-8))

    if self.is_train:
      return cal_weight(features[edges[:, 0]], features[edges[:, 1]], self.tau,
                        1.0)
    else:
      return cal_weight(features[edges[:, 0]], features[edges[:, 1]], self.tau,
                        3.0)

  def propagation(self, point_visible_mask):
    visible_edges_mask = point_visible_mask[self.edges].any(dim=1)  # (m)
    visible_edges = self.edges[visible_edges_mask]  # (m, 2)

    edge_weight = self.cal_edge_weight(visible_edges)

    return visible_edges, edge_weight

  def propagation_loss(self, point_visible_mask):
    if not self.is_graph_exist():
      return 0.0
    edges_selected, edge_weight = self.propagation(
        point_visible_mask)  # (m, 2), (m, 1)
    if edge_weight.shape[0] == 0:
      return 0.0

    self.logger.debug(
        f"ew max {edge_weight.max()} min {edge_weight.min()} mean {edge_weight.mean()}"
    )

    features = self.model.segment_activation(self.model._feature)
    af0 = features[edges_selected[:, 0]]  # (m, f)
    af1 = features[edges_selected[:, 1]]  # (m, f)

    return ((edge_weight) * (af0 - af1)**2).sum()


# Graph Helper with Bilateral Graph Normalized Laplacian
class GraphHelper_BilateralGraphNormLaplacian(GraphBaseClass):

  def __init__(self, logger, gaussian_model: GaussianModel):
    super(GraphHelper_BilateralGraphNormLaplacian,
          self).__init__(logger, gaussian_model)
    self.logger = logger

    self.tau = 0.05
    # self.mlp = None
    self.spatial_sigma_scale = 1.
    self.is_train = True

  def init_model(self):
    self.logger.info("This class is not neural part.")

  def train(self):
    self.is_train = True

  def eval(self):
    self.is_train = False

  def step(self, iteration):
    pass

  def save(self, path):
    super().save(path)

  def load(self, path):
    super().load(path)

  def cal_edge_weight(self, edges):

    self.logger.info("edges shape: {}".format(edges.shape))
    # init
    points = self.model.scatter_mean_to_anchor(
        self.model.get_xyz).detach()  # (n, 3)
    features = self.model.segment_activation(
        self.model._feature).detach()  # (n, f)

    if self.is_train:

      def cal_weight(v0, v1, sigma, scale):
        return torch.exp(-torch.sum(
            (scale * (v0 - v1))**2, dim=1, keepdim=True) /
                         (2 * sigma**2 + 1e-8))

      return cal_weight(features[edges[:, 0]], features[edges[:, 1]], self.tau,
                        1.0)
    else:

      def cal_weight(v0, v1, sigma, scale):
        return torch.exp(-torch.sum(
            (scale * (v0 - v1))**2, dim=1, keepdim=True) /
                         (2 * sigma**2 + 1e-8))

      return cal_weight(features[edges[:, 0]], features[edges[:, 1]], self.tau,
                        3.0)

  def propagation(self, point_visible_mask):
    # get visible points id
    # visible_points_id = torch.nonzero(point_visible_mask)  # (n)

    # get the edges contain visible points
    visible_edges_mask = point_visible_mask[self.edges].any(dim=1)  # (m)
    visible_edges = self.edges[visible_edges_mask]  # (m, 2)

    edge_weight = self.cal_edge_weight(visible_edges)

    return visible_edges, edge_weight

  def propagation_loss(self, point_visible_mask):
    if not self.is_graph_exist():
      return 0.0
    edges_selected, edge_weight = self.propagation(
        point_visible_mask)  # (m, 2), (m, 1)
    if edge_weight.shape[0] == 0:
      return 0.0

    self.logger.debug(
        f"ew max {edge_weight.max()} min {edge_weight.min()} mean {edge_weight.mean()}"
    )

    features = self.model.segment_activation(self.model._feature)
    af0 = features[edges_selected[:, 0]]  # (m, f)
    af1 = features[edges_selected[:, 1]]  # (m, f)

    # laplacian_loss = ((edge_weight) * (af0 - af1)**2).sum()
    # return laplacian_loss
    """
    We have provided a more stable graph propagation implementation!
    The normalized Laplacian can effectively avoid problems caused by node degrees.
    """

    deg = torch_scatter.scatter_add(edge_weight.squeeze(1), edges_selected[:,0].long(), dim=0, dim_size=features.size(0)) \
    + torch_scatter.scatter_add(edge_weight.squeeze(1), edges_selected[:,1].long(), dim=0, dim_size=features.size(0))
    norm_w = edge_weight / (
        (deg[edges_selected[:, 0]] *
         deg[edges_selected[:, 1]]).sqrt().unsqueeze(1) + 1e-6)
    norm_laplacian_loss = (norm_w * (af0 - af1).pow(2)).sum()
    return norm_laplacian_loss


# Choose which GraphHelper to use
# GraphHelper = GraphHelper_GNNGraphProp
# GraphHelper = GraphHelper_SigmoidGraphLaplacian
# GraphHelper = GraphHelper_BilateralGraphLaplacian
GraphHelper = GraphHelper_BilateralGraphNormLaplacian
# GraphHelper = GraphHelper_ScatterLoss
