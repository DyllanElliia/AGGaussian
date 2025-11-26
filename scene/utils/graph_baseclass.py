import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

from einops import rearrange, repeat

from tqdm import tqdm
from itertools import product

import torch_scatter
import pytorch3d.ops

from .. import GaussianModel


class GraphBaseClass_v1(nn.Module):

  def __init__(self, logger, model: GaussianModel):
    super(GraphBaseClass_v1, self).__init__()
    self.logger = logger
    self.model = model
    self.edges = None

    self.is_graph_exist_ = False

    self.neighbor_offsets = torch.tensor(list(product([-1, 0, 1], repeat=3)),
                                         dtype=torch.int16,
                                         device="cuda")  # (27, 3)
    self.hash_32 = torch.tensor([73856093, 19349663, 83492791],
                                dtype=torch.int32,
                                device="cuda")

    self.find_radii = True
    self.graph_query_chunk = 4096
    self.max_knn_neighbors = 128

  def step(self, iteration):
    pass

  def train(self):
    pass

  def eval(self):
    pass

  def save(self, path):
    self.logger.info(f"Save graph to {path}")
    torch.save(self.edges, path)

  def load(self, path):
    self.logger.info(f"Load graph from {path}")
    try:
      self.edges = torch.load(path, map_location=self.model._anchor_xyz.device)
      self.set_graph_exist(True)
    except Exception as e:
      self.logger.error(f"Load graph failed: {e}")

  def is_graph_exist(self):
    if self.edges is None:
      self.is_graph_exist_ = False

    return self.is_graph_exist_

  def set_graph_exist(self, is_exist):
    self.is_graph_exist_ = is_exist
    if not is_exist:
      self.clean_graph()

  def clean_graph(self):
    self.edges = None
    if "edge_weights" in self.__dict__:
      del self.edge_weights
    torch.cuda.empty_cache()
    self.is_graph_exist_ = False

  def simplify_graph(self, max_edges_num=10, remove_edges=False):
    # this function is used to simplify the graph, remove the edges and save the max_edges_num shortest edges
    if self.edges is None:
      return

    # compute the anchor edge num
    anchor_edge_num = self.compute_anchor_edge_num().int()
    # select the anchor with edge num > max_edges_num
    selected_anchor = anchor_edge_num > max_edges_num

    # split the edges to selected and saved
    select_cond = selected_anchor[self.edges[:, 0]] | selected_anchor[
        self.edges[:, 1]]
    selected_edges = self.edges[select_cond]
    saved_edges = self.edges[~select_cond]

    self.edges = saved_edges

    if remove_edges:
      return

    self.logger.info(
        f"Graph simplified with {selected_edges.shape[0]} edges selected.")

    selected_anchor_num = selected_anchor.sum()

    # compute the anchor id
    anchor_id = torch.arange(selected_anchor.shape[0],
                             device="cuda")[selected_anchor]
    anchor_num = anchor_edge_num[selected_anchor]

    # sort the anchor by edge num, from large to small
    _, sorted_anchor = torch.sort(anchor_num, descending=True)
    anchor_id = anchor_id[sorted_anchor]

    self.logger.info(
        f"Select {selected_anchor_num} anchors with edge num > {max_edges_num}")

    points = self.model._anchor_xyz.data.contiguous().half()

    for i in tqdm(range(selected_anchor_num), desc="Simplify graph"):
      a_id = anchor_id[i]

      # select the edges contains the anchor
      select_cond = (selected_edges[:, 0] == a_id) | (selected_edges[:, 1]
                                                      == a_id)
      anchor_edges = selected_edges[select_cond]

      # remove the anchor_edges
      selected_edges = selected_edges[~select_cond]

      if anchor_edges.shape[0] <= max_edges_num:

        self.edges = torch.cat([self.edges, anchor_edges], dim=0)
        continue

      # self.logger.info(
      #     f"Select {anchor_edges.shape[0]} edges for anchor {a_id}")

      # compute the distance between the anchor and other points
      anchor_edges_dist = torch.norm(points[anchor_edges[:, 0]] -
                                     points[anchor_edges[:, 1]],
                                     dim=1)
      # get top max_edges_num shortest edges
      _, top_idx = torch.topk(anchor_edges_dist, max_edges_num, largest=False)

      # save the top edges
      self.edges = torch.cat([self.edges, anchor_edges[top_idx]], dim=0)

  def link_isolated_anchor(self, points, features):
    # this function is used to link the isolated anchor
    # the isolated anchor is the anchor with no edges
    # the isolated anchor will be linked to the nearest anchor
    if self.edges is None:
      return

    anchor_edge_num = self.compute_anchor_edge_num().int()
    isolated_anchor = anchor_edge_num == 0

    if isolated_anchor.sum() == 0:
      return

    self.logger.info(
        f"Link {isolated_anchor.sum()} isolated anchors to the nearest anchor")

    pf = torch.cat([points, features], dim=1)

    isolated_anchor_id = torch.arange(
        points.shape[0], device="cuda")[isolated_anchor]  # (isolated_num,)
    anchor_id = torch.arange(points.shape[0],
                             device="cuda")[~isolated_anchor]  # (anchor_num,)

    # link the isolated anchor to the nearest anchor via pytorch3d
    isolated_pf = pf[isolated_anchor_id]  # (isolated_num, 3+f)
    anchor_pf = pf[anchor_id]  # (anchor_num, 3+f)

    _, idx, _ = pytorch3d.ops.knn_points(
        isolated_pf.unsqueeze(0),
        anchor_pf.unsqueeze(0),
        K=1,
        return_sorted=False)  # (1, isolated_num, 1)
    idx = idx.reshape(-1)  # (isolated_num,)

    # create the edges
    e = torch.stack([isolated_anchor_id, anchor_id[idx]],
                    dim=1)  # (isolated_num, 2)
    # sort the idx, make sure the second id is larget than the first id
    e = torch.sort(e, dim=1)[0]
    self.edges = torch.cat([self.edges, e], dim=0)

  def coord2hash(self, coords):
    # coords: torch.intTensor(n,3), int16
    return torch.sum(coords * self.hash_32, dim=1)

  def compute_hash_id_of_points(self, points, voxel_size):
    # points: (n,3)
    # return: (n,)
    voxel_coords = (points / voxel_size).type(torch.int32)
    hash_id = self.coord2hash(voxel_coords)
    return voxel_coords, hash_id

  def get_neighbour_coords(self, coords):
    # coords: torch.intTensor(n,3)
    # return: torch.intTensor(n,27,3)
    return coords.unsqueeze(1) + self.neighbor_offsets.unsqueeze(0)

  def build_edges(self,
                  start_idx,
                  end_idx,
                  points,
                  coords_and_hash,
                  radii,
                  device,
                  dis_threshold=None,
                  edge_dist_scale=1.0):
    if start_idx == end_idx:
      return torch.empty((0, 2), dtype=torch.int32, device=device)

    n = points.shape[0]

    # calculate all pairs of points, ensure j > i
    idx_i = torch.arange(start_idx, end_idx, dtype=torch.int32, device=device)
    idx_j = torch.arange(n, dtype=torch.int32, device=device)

    # get hash id of points
    voxel_coords, hash_id = coords_and_hash
    voxel_coords_i = voxel_coords[idx_i]  # (end - start, 3)
    voxel_i_neighbour_coords = self.get_neighbour_coords(
        voxel_coords_i)  # (end - start, 27, 3)

    hash_i_neighbour = self.coord2hash(voxel_i_neighbour_coords.view(
        -1, 3)).view(idx_i.shape[0], -1)  # (end - start, 27)

    # filter the idx_j which has the same hash id in hash_i_neighbour
    hash_j_expand = hash_id.unsqueeze(0).repeat(end_idx - start_idx, 1)
    mask = (hash_j_expand.unsqueeze(-1) == hash_i_neighbour.unsqueeze(1)).any(
        dim=-1)

    idx_i = idx_i.unsqueeze(1).repeat(1, n)  # (end - start, n)
    idx_j = idx_j.unsqueeze(0).repeat(end_idx - start_idx,
                                      1)  # (end - start, n)

    # select pairs with j > i
    mask = mask & (idx_j > idx_i)  # (num_pairs, n)
    idx_i = idx_i[mask]  # (num_pairs)
    idx_j = idx_j[mask]  # (num_pairs)

    # get selected points and radii
    selected_points_i = points[idx_i]  # (num_pairs, 3)
    selected_points_j = points[idx_j]  # (num_pairs, 3)
    selected_radii_i = radii[idx_i].squeeze(1)  # (num_pairs)
    selected_radii_j = radii[idx_j].squeeze(1)  # (num_pairs)

    # calculate squared distance
    dis_squared = torch.sum((selected_points_i - selected_points_j)**2,
                            dim=1)  # (num_pairs)

    # calculate squared sum of radii
    # ------------------------------------------------
    # max_r = torch.min(selected_radii_i, selected_radii_j) * 1.5
    # sum_radii_squared = edge_dist_scale * (torch.min(
    #     selected_radii_i, max_r) + torch.min(selected_radii_j, max_r))**2
    # ------------------------------------------------
    # sum_radii_squared = (2. * torch.min(selected_radii_i, selected_radii_j))**2
    # ------------------------------------------------
    # sum_radii_squared = (dis_threshold)**2
    if self.use_radii:
      max_r = torch.min(selected_radii_i, selected_radii_j) * 1.5
      sum_radii_squared = edge_dist_scale * (torch.min(
          selected_radii_i, max_r) + torch.min(selected_radii_j, max_r))**2
    else:
      sum_radii_squared = (dis_threshold)**2

    # select pairs with distance < sum of radii
    mask = (dis_squared < sum_radii_squared)
    # if dis_threshold is not None:
    #   mask = mask & (dis_squared < dis_threshold)
    mask = mask & (selected_radii_i < dis_threshold)

    idx_i = idx_i[mask]  # (num_valid_pairs)
    idx_j = idx_j[mask]  # (num_valid_pairs)

    return torch.stack((idx_i, idx_j), dim=1)  # (num_valid_pairs, 2)

  def _build_graph_with_pytorch3d(self, points, radii, dis_threshold):
    """Build graph edges with chunked knn queries from PyTorch3D."""
    device = points.device
    n = points.shape[0]
    if n < 2:
      return torch.empty((0, 2), dtype=torch.int32, device=device)

    max_knn = max(1, min(int(self.max_knn_neighbors), n - 1))
    query_chunk = max(1, int(self.graph_query_chunk))
    dis_threshold_sq = float(dis_threshold)**2
    all_points = points.unsqueeze(0)
    radii_1d = radii.squeeze(1)

    chunk_edges = []
    for start in tqdm(range(0, n, query_chunk), desc="Building graph (knn)"):
      end = min(start + query_chunk, n)
      chunk = points[start:end].unsqueeze(0)

      knn = pytorch3d.ops.knn_points(chunk,
                                     all_points,
                                     K=max_knn,
                                     return_nn=False,
                                     return_sorted=False)

      idx = knn.idx[0]  # (chunk, K)
      dists = knn.dists[0]  # (chunk, K)

      query_ids = torch.arange(start, end, device=device,
                               dtype=torch.int64).unsqueeze(1).expand_as(idx)
      neighbor_ids = idx
      upper_mask = neighbor_ids > query_ids
      if not upper_mask.any():
        continue

      neighbor_ids = neighbor_ids[upper_mask]
      query_ids = query_ids[upper_mask]
      dists = dists[upper_mask]

      radii_i = radii_1d[query_ids]
      radii_j = radii_1d[neighbor_ids]
      if self.use_radii:
        max_r = torch.min(radii_i, radii_j) * 1.5
        sum_radii_sq = self.edge_dist_scale * (torch.min(radii_i, max_r) +
                                               torch.min(radii_j, max_r))**2
      else:
        sum_radii_sq = dis_threshold_sq

      valid = dists < sum_radii_sq
      valid = valid & (radii_i < dis_threshold)
      if not valid.any():
        continue

      src = query_ids[valid].to(torch.int32)
      dst = neighbor_ids[valid].to(torch.int32)
      edge_ij = torch.stack((src, dst), dim=1)
      edge_ij = torch.sort(edge_ij, dim=1)[0]
      chunk_edges.append(edge_ij.cpu())

    if len(chunk_edges) == 0:
      return torch.empty((0, 2), dtype=torch.int32, device=device)

    return torch.cat(chunk_edges, dim=0).to(device=device)

  def _build_graph_with_hash(self, points, radii, voxel_threshold,
                             dis_threshold):
    self.logger.info("Fallback to hash-based graph builder.")
    coords_and_hashs = self.compute_hash_id_of_points(points, voxel_threshold)
    coords, _ = coords_and_hashs
    self.logger.info(f"min max coords: {coords.min(dim=0)} {coords.max(dim=0)}")
    self.logger.info(f"Dis threshold: {dis_threshold}")
    n = points.shape[0]
    device = points.device

    done = False
    while done == False:
      try:
        torch.cuda.empty_cache()
        edges = []
        for start in tqdm(range(0, n, self.batch_size),
                          desc="Building graph (hash)"):
          end = min(start + self.batch_size, n)
          edge = self.build_edges(start, end, points, coords_and_hashs, radii,
                                  device, dis_threshold, None)

          if edge.shape[0] > 0:
            edges.append(edge.cpu())
        done = True
      except Exception as e:
        self.logger.warning(
            f"Batch_size {self.batch_size} failed with error: {e}. Shrinking batch."
        )
        self.batch_size = max(1, int(self.batch_size / 2))
        self.logger.warning(f"New Batch_size is {self.batch_size}")

    torch.cuda.empty_cache()
    self.logger.info("Merge edges...")
    for i in range(len(edges)):
      edges[i] = edges[i].to(dtype=torch.int32, device=device)

    if len(edges) > 0:
      merged = torch.cat(edges, dim=0)
      return torch.sort(merged, dim=1)[0]
    return torch.empty((0, 2), dtype=torch.int32, device=device)

  def build_graph(self,
                  edge_dist_scale=1.0,
                  debug_func=None,
                  link_isolated=False,
                  use_radii=False):
    """
    Hi! We have updated a new build_graph that is super fast and memory efficient!
    Feel free to use it in larger scenes! :) (Whisper: Except for autonomous driving)
    Anchor-Graph with LOD is all you need!
    """
    self.logger.info("Building graph... in GraphBaseClass")
    self.edge_dist_scale = edge_dist_scale
    self.use_radii = use_radii
    with torch.no_grad():
      # if graph is exist, deleting it
      if self.is_graph_exist():
        self.clean_graph()

      self.batch_size = self.graph_query_chunk

      # init
      points = self.model._anchor_xyz.data.contiguous().half()
      if False:
        radii = self.model.scatter_mean_to_anchor(
            (self.model.get_xyz - points[self.model._point2anchor]).norm(
                dim=-1, keepdim=True)).data.contiguous().half()

      else:
        if debug_func is not None:
          radii = debug_func(self.model)
        else:
          radii = self.model._anchor_radii.data.contiguous().half()
      radii = radii.float()
      self.logger.info(
          f"Mean radius {radii.shape} of anchor points : {radii.mean().item()}")
      self.logger.info(
          f"Min and Max radius of anchor points : {radii.min().item()} {radii.max().item()}"
      )
      device = points.device
      n = points.shape[0]

      # TODO: not percent_dense, check out!
      # dis_threshold = self.model.percent_dense * 2.3
      base_dis_threshold = self.model.get_voxel_size(
      ) * self.model.update_init_factor * self.edge_dist_scale
      dis_threshold = base_dis_threshold + 1e-4

      torch.cuda.empty_cache()
      edges = None
      try:
        points_fp32 = points.float()
        edges = self._build_graph_with_pytorch3d(points_fp32, radii,
                                                 dis_threshold)
        del points_fp32
      except Exception as e:
        self.logger.warning(
            f"PyTorch3D knn graph build failed ({e}). Will fallback to hash search."
        )

      if edges is None:
        points_fp32 = points.float()
        edges = self._build_graph_with_hash(points_fp32, radii,
                                            base_dis_threshold, dis_threshold)
        del points_fp32

      if edges.numel() > 0:
        edges = torch.sort(edges, dim=1)[0]
        edges = torch.unique(edges, dim=0)

      self.edges = edges
      self.logger.info(f"Graph built with {self.edges.shape[0]} edges.")
      # self.edges = torch.unique(self.edges, dim=0)
      # self.logger.info(f"Graph built with {self.edges.shape[0]} unique edges.")
      self.logger.info(
          f"Edges need {self.edges.element_size() * self.edges.nelement() / 1024**3:.2f}GB gpu memory."
      )

      # torch.cuda.empty_cache()
      # self.simplify_graph(remove_edges=False, max_edges_num=10)
      torch.cuda.empty_cache()
      if link_isolated:
        self.link_isolated_anchor(
            points, self.model.segment_activation(self.model._feature))
        torch.cuda.empty_cache()
      self.set_graph_exist(True)

  def init_model(self):
    pass

  def get_boundary_anchor(self, selected_anchor):
    """
    高效计算边界锚点(显存友好版)
    参数:
        selected_anchor: (n_anchors,) bool tensor - 已选中的锚点mask
    返回:
        boundary_mask: (n_anchors,) bool tensor - 边界锚点mask
    """

    # 步骤2：向量化判断边的端点状态
    # 获取每条边两个端点的选中状态 (E, 2)
    edge_states = selected_anchor[self.edges]

    # 步骤3：找出只有一个端点被选中的边（边界边）
    # 异或运算：a在选中集且b不在，或b在选中集且a不在
    is_boundary_edge = edge_states[:, 0] ^ edge_states[:, 1]  # (E,)

    # 步骤4：提取边界边中的选中端点
    boundary_edges = self.edges[is_boundary_edge]  # (B, 2)
    # boundary_anchors = boundary_edges[edge_states[is_boundary_edge]]  # (B,)

    # 步骤5：生成边界mask（与selected_anchor求交）
    boundary_mask = torch.zeros_like(selected_anchor)
    # boundary_mask[boundary_anchors.unique()] = True
    boundary_mask[boundary_edges.flatten()] = True
    # return boundary_mask & selected_anchor  # 确保结果在选中集内
    return boundary_mask

  def select_edges_via_graph_diffusion_(
      self,
      init_anchor_id,
      edge_weight_mask,
      #  feature_threshold=0.1,
      iter_num=10,
      small_anchor_mask=None,
      return_boundary_anchor=False,
  ):
    selected_edges = torch.zeros(self.edges.shape[0],
                                 dtype=torch.bool,
                                 device="cuda")
    selected_anchor = torch.zeros(self.model._feature.shape[0],
                                  dtype=torch.bool,
                                  device="cuda")
    # if big_anchor_mask is None:
    #   big_anchor_mask = self.compute_anchor_edge_num().int()
    #   big_anchor_mask = big_anchor_mask < 20

    features = self.model.segment_activation(self.model._feature)  # (n, f)
    # edges_features_weight = torch.norm(features[self.edges[:, 0]] -
    #                                    features[self.edges[:, 1]],
    #                                    dim=1)  # (e,)
    # edges_features_weight = edges_features_weight < feature_threshold

    curr_anchor_mask = torch.zeros(features.shape[0],
                                   dtype=torch.bool,
                                   device="cuda")
    curr_anchor_mask[init_anchor_id] = True

    for i in range(iter_num):
      # select edges contains current anchor
      select_edges_mask = curr_anchor_mask[self.edges].any(dim=1)
      # remove the edges that already selected
      select_edges_mask = select_edges_mask & ~selected_edges
      # remove the edges that contains the selected anchor
      select_edges_mask = select_edges_mask & ~selected_anchor[self.edges].any(
          dim=1)
      # select the edges with feature_weight less than threshold
      select_edges_mask = select_edges_mask & edge_weight_mask
      # append the select edges to the selected edges
      selected_edges = selected_edges | select_edges_mask

      # update the selected anchor
      selected_anchor = selected_anchor | curr_anchor_mask
      # update the current anchor
      curr_anchor_mask[self.edges[select_edges_mask]] = True
      # remove the selected anchor and big anchor
      curr_anchor_mask = curr_anchor_mask & ~selected_anchor
      if small_anchor_mask is not None:
        curr_anchor_mask = curr_anchor_mask & small_anchor_mask
      self.logger.info(
          f"iter {i} selected {selected_edges.sum()} edges, {curr_anchor_mask.sum()} anchors"
      )
      if curr_anchor_mask.sum() == 0:
        self.logger.info("no more anchor")
        break
    if return_boundary_anchor:
      boundary_mask = self.get_boundary_anchor(selected_anchor)
      return selected_edges, selected_anchor, boundary_mask
    else:
      return selected_edges, selected_anchor

  def select_edges(self,
                   init_anchor_id,
                   edge_weight_threshold=0.5,
                   iter_num=20,
                   edge_select_condition=lambda ew, threshold: ew > threshold,
                   big_anchor_mask=None,
                   return_boundary_anchor=False):
    if not self.is_graph_exist():
      self.logger.error("Graph is not exist")
      return None, None

    if not "edge_weights" in self.__dict__:
      self.edge_weights = self.compute_all_edge_weight(
          cal_edge_func=lambda e: self.cal_edge_weight(e)).squeeze(1)

    # edge_mask = self.edges_weight > edge_weight_threshold
    edge_mask = edge_select_condition(self.edge_weights, edge_weight_threshold)
    if big_anchor_mask is None:
      big_anchor_mask = torch.zeros_like(self.model._anchor_radii,
                                         dtype=torch.bool).squeeze(1)

    # return [selected_edges, selected_anchor, (if return_boundary_anchor) boundary_anchor]
    return self.select_edges_via_graph_diffusion_(init_anchor_id, edge_mask,
                                                  iter_num, ~big_anchor_mask,
                                                  return_boundary_anchor)

  def cal_edge_weight(self, edges):
    raise NotImplementedError

  def compute_all_edge_weight(self, cal_edge_func):
    with torch.no_grad():
      self.logger.info("Computing all edge weight...")
      edge_weight = torch.empty((self.edges.shape[0], 1),
                                dtype=torch.float32,
                                device=self.model._anchor_xyz.device)
      # batch_size = 2**14

      # # self.logger.info("Using sigmoid weight...")
      # # cal_edge_func = self.cal_edge_bilateral_weight

      # for start in tqdm(range(0, self.edges.shape[0], batch_size),
      #                   desc="Computing edge weight"):
      #   end = min(start + batch_size, self.edges.shape[0])

      #   edge_weight[start:end] = cal_edge_func(self.edges[start:end])
      edge_weight = cal_edge_func(self.edges)

      return edge_weight

  def compute_all_edge_feature_dist(self, edges=None):
    with torch.no_grad():
      self.logger.info("Computing all edge feature dist...")
      features = self.model.segment_activation(self.model._feature)
      f0 = features[self.edges[:, 0]]
      f1 = features[self.edges[:, 1]]

      return torch.norm(f0 - f1, dim=1)

  def compute_anchor_edge_num(self):
    with torch.no_grad():
      self.logger.info("Computing anchor edge num...")
      anchor_edge_num = torch.zeros(self.model._feature.shape[0],
                                    dtype=torch.int64,
                                    device="cuda")
      flatten_edges = self.edges.flatten().long()
      anchor_edge_num = anchor_edge_num.scatter_add(
          0, flatten_edges, torch.ones_like(flatten_edges))
      return anchor_edge_num

  def get_super_anchor(self, options):
    # from .utils.anchor_merger import AnchorMerger_MeanShiftUnionFind
    from .utils.anchor_merger_w_post import AnchorMerger_MeanShiftUnionFind
    with torch.no_grad():
      cluster = AnchorMerger_MeanShiftUnionFind(
          options.merge_bandwidth,
          options.merge_tol,
          shift_vec_func=lambda x: x / (x.norm(dim=1, keepdim=True) + 1e-5),
          logger=self.logger)
      cluster.init_peak(Feat=self.model.segment_activation(self.model._feature),
                        Pos=self.model._anchor_xyz,
                        Edge=self.edges.transpose(0, 1))
      cluster_id, mask, peak_feat = cluster.run(
          iter_max=50,
          shifted_tol=1e-5,
          fragment_search_radius=self.model.voxel_size *
          self.model.update_init_factor * self.edge_dist_scale * 8.0)
      self.model._feature = peak_feat.float()

      new_anchor_num = cluster_id.max() + 1

      super_anchor_xyz = torch.zeros((new_anchor_num, 3), device="cuda")
      super_anchor_feature = torch.zeros(
          (new_anchor_num, self.model._feature.shape[1]), device="cuda")
      torch_scatter.scatter_mean(self.model._anchor_xyz,
                                 cluster_id,
                                 dim=0,
                                 out=super_anchor_xyz)
      torch_scatter.scatter_mean(cluster.peak_f.float(),
                                 cluster_id,
                                 dim=0,
                                 out=super_anchor_feature)
      super_anchor_edge = self.edges.clone()
      super_anchor_edge[:, 0] = cluster_id[self.edges[:, 0]]
      super_anchor_edge[:, 1] = cluster_id[self.edges[:, 1]]
      cluster.close()
      del cluster
      # make sure the edge is sorted
      super_anchor_edge = torch.sort(super_anchor_edge, dim=1)[0]
      # union the edges
      super_anchor_edge = torch.unique(super_anchor_edge, dim=0)  # (e, 2)
      return super_anchor_xyz, super_anchor_feature, super_anchor_edge, cluster_id


GraphBaseClass = GraphBaseClass_v1
