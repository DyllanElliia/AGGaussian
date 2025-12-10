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


class AnchorMerger_MeanShiftUnionFind:

  def __init__(self,
               bandwidth: float,
               tol: float,
               weight_mode='gaussian',
               shift_vec_func: nn.Module = lambda x: x,
               logger=None):
    self.bandwidth = bandwidth
    self.tol = tol
    self.weight_mode = weight_mode
    self.shift_vec_func = shift_vec_func
    self.logger = logger

    # Validate the weight_mode parameter
    assert self.weight_mode in [
        'ball', 'gaussian'
    ], "weight_mode must be either 'ball' or 'gaussian'"
    # Define weight computation and shift vector functions based on weight_mode using a mapping
    weight_functions = {
        'ball':  # Hard bandwidth mask: only consider points within the bandwidth
            lambda distance_sq: distance_sq < self.bandwidth**2,  # compute_weight
        'gaussian':  # Gaussian kernel weights: weight points based on their distance
            lambda distance_sq: torch.exp(-distance_sq / (2 * self.bandwidth**2)),  # compute_weight
    }

    # Assign the corresponding functions based on weight_mode
    self.compute_weight = weight_functions[self.weight_mode]
    # -------------------------------------------- #
    # Peak_f: [N, f] (N: number of points, f: feature dimension)
    self.peak_f = None
    # Peak_edge: [2, E] (E: number of edges)
    self.peak_edge = None
    # # Org_f_2_peak: [N] (N: number of points) mapping from original points to peak points
    # self.org_f_2_peak = None
    self.f = None

  def init_peak(self, Feat, Pos, Edge):
    """
    Initialize the peak points and edges from the input graph
      Feat: [N, f] (N: number of points, f: feature dimension)
      Edge: [2, E] (E: number of edges)
    """
    self.logger.info("Initializing peak points and edges")
    self.init_feat = Feat.clone().half()
    self.init_xyz = Pos.clone().half()
    # Initialize the peak points and edges
    self.peak_f = Feat.clone().half()
    self.peak_edge = Edge.clone().long()
    # self.org_f_2_peak = torch.range(0, Feat.shape[0] - 1).long().to(Feat.device)
    self.f = Feat.clone().half()

  def close(self):
    self.peak_f = None
    self.peak_edge = None
    self.f = None
    torch.cuda.empty_cache()

  def step(self):
    """
    Perform one iteration of the mean shift algorithm and graph
    """
    self_m = torch.arange(self.peak_f.shape[0], device=self.peak_f.device)

    # Compute the pairwise distances between the peak point and original points
    d2 = torch.cat([
        (self.peak_f[self.peak_edge[0]] - self.f[self.peak_edge[1]]).norm(
            p=2, dim=1),
        (self.peak_f[self.peak_edge[1]] - self.f[self.peak_edge[0]]).norm(
            p=2, dim=1),
        (self.peak_f[self_m] - self.f[self_m]).norm(p=2, dim=1),
    ],
                   dim=0)  # (3E, )
    indices = torch.cat([self.peak_edge[0], self.peak_edge[1], self_m],
                        dim=0).long()  # (3E, )
    # Compute the weights based on the pairwise distances
    d2 = self.compute_weight(d2).unsqueeze(1)  # (3E, 1)
    d2f = d2 * torch.cat(
        [self.f[self.peak_edge[1]], self.f[self.peak_edge[0]], self.f[self_m]],
        dim=0)  # (3E, f)

    # compute shifted
    shifted = torch.zeros(self.f.shape[0],
                          self.f.shape[1] + 1,
                          device=self.f.device,
                          dtype=self.f.dtype)  # (N, f+1)
    torch_scatter.scatter_add(torch.cat([d2f, d2], dim=1),
                              indices,
                              dim=0,
                              out=shifted)  # (N, f+1)
    shifted = shifted[:, :-1] / (shifted[:, -1].unsqueeze(1) + 1e-5)  # (N, f)

    shifted = self.shift_vec_func(shifted)

    shifted_dist = (shifted - self.peak_f).norm(p=2, dim=1)

    return shifted, shifted_dist

  def merge(self):
    # merge the peak points based on the graph and the edge weights
    self.logger.info("Merging the peak points")

    def filter_edges_by_distance(p, edges, tol):
      dis = (p[edges[0]] - p[edges[1]]).norm(dim=1)
      mask = dis < tol
      return edges[:, mask]

    def union_find_merge(edges, n):
      # Init: each point is its own parent
      parent = torch.arange(n, device=edges.device)

      # Find: find the root parent of each point
      prev_parent = None
      while True:
        # 1. Update the parent of each point
        parent = parent[parent]

        # 2. Find the different parent pairs
        p0 = parent[edges[0]]
        p1 = parent[edges[1]]
        diff_mask = p0 != p1
        # 3. Merge the different parent pairs into the same parent
        if diff_mask.any():
          min_p = torch.min(p0[diff_mask], p1[diff_mask])
          parent[p0[diff_mask]] = min_p
          parent[p1[diff_mask]] = min_p

        # 4. Check if the parent is the same as the previous parent
        if prev_parent is not None and torch.equal(parent, prev_parent):
          break

        prev_parent = parent.clone()

      # Update one more time, to make sure all the parent is the root parent
      return parent[parent]

    def get_cluster_id_from_parent(parent):
      _, cluster_id = torch.unique(parent, return_inverse=True)
      return cluster_id

    edges_filtered = filter_edges_by_distance(self.peak_f, self.peak_edge,
                                              self.tol)
    # mask the points which are be selected
    mask = torch.zeros(self.peak_f.shape[0],
                       dtype=torch.bool,
                       device=self.peak_f.device)
    mask[edges_filtered.flatten()] = True

    n = self.peak_f.shape[0]
    if edges_filtered.numel() == 0:
      return torch.arange(n, device=self.peak_f.device), mask

    parent = union_find_merge(edges_filtered, n)

    # get the cluster id, which is the root parent of merged points
    cluster_id = get_cluster_id_from_parent(parent)
    return cluster_id, mask

  def merge_fragments(self, cluster_id, mask, min_cluster_size=10, radius=None):
    """Post-process small clusters by attaching them to nearby large ones."""
    if min_cluster_size is None or min_cluster_size <= 0:
      return cluster_id, mask

    device = cluster_id.device
    num_clusters = int(cluster_id.max().item()) + 1
    if num_clusters == 0:
      return cluster_id, mask
    print(f"Number of clusters before fragment merging: {num_clusters}")
    cluster_sizes = torch.bincount(cluster_id, minlength=num_clusters)
    big_cluster_mask = cluster_sizes >= min_cluster_size
    if big_cluster_mask.all():
      return cluster_id, mask
    if big_cluster_mask.sum() == 0:
      return cluster_id, mask

    small_cluster_mask = ~big_cluster_mask
    radius = float(radius) if radius is not None else float(self.bandwidth)
    if radius <= 0:
      radius = None

    cluster_feat = torch_scatter.scatter_mean(self.init_feat.float(),
                                              cluster_id,
                                              dim=0,
                                              dim_size=num_clusters)
    cluster_pos = torch_scatter.scatter_mean(self.init_xyz.float(),
                                             cluster_id,
                                             dim=0,
                                             dim_size=num_clusters)

    reassignment = torch.full((num_clusters,),
                              -1,
                              dtype=torch.long,
                              device=device)

    if self.peak_edge.numel() > 0:
      src = self.peak_edge[0].long()
      dst = self.peak_edge[1].long()
      edge_dist = (self.peak_f[src] - self.peak_f[dst]).norm(dim=1)
      directed_src = torch.cat([src, dst], dim=0)
      directed_dst = torch.cat([dst, src], dim=0)
      directed_dist = torch.cat([edge_dist, edge_dist], dim=0)

      src_clusters = cluster_id[directed_src]
      dst_clusters = cluster_id[directed_dst]
      candidate_mask = small_cluster_mask[src_clusters] & big_cluster_mask[
          dst_clusters]

      if candidate_mask.any():
        candidate_small = src_clusters[candidate_mask]
        candidate_big = dst_clusters[candidate_mask]
        candidate_dist = directed_dist[candidate_mask]
        _, argmin = torch_scatter.scatter_min(candidate_dist,
                                              candidate_small,
                                              dim_size=num_clusters)
        valid = (argmin >= 0) & (argmin < candidate_big.shape[0])
        if valid.any():
          reassignment[valid] = candidate_big[argmin[valid]]

    small_ids = torch.nonzero(small_cluster_mask, as_tuple=False).flatten()
    need_graph = small_ids[reassignment[small_ids] < 0]

    if need_graph.numel() > 0:
      big_point_mask = big_cluster_mask[cluster_id]
      if big_point_mask.any():
        small_centers = cluster_pos[need_graph]
        big_points = self.init_xyz[big_point_mask].float()
        big_point_clusters = cluster_id[big_point_mask]
        if small_centers.numel() > 0 and big_points.numel() > 0:
          # Prefer spatial neighbors by querying big-cluster points in the original graph space
          knn = pytorch3d.ops.knn_points(
              small_centers.unsqueeze(0),
              big_points.unsqueeze(0),
              K=1,
              return_nn=False,
              return_sorted=False,
          )
          point_dist = knn.dists.squeeze(0).squeeze(-1).sqrt()
          point_idx = knn.idx.squeeze(0).squeeze(-1)
          valid = point_idx >= 0
          if radius is not None:
            valid = valid & (point_dist <= radius)
          if valid.any():
            reassignment[need_graph[valid]] = big_point_clusters[
                point_idx[valid]]

    need_radius = small_ids[reassignment[small_ids] < 0]
    if radius is not None and need_radius.numel() > 0:
      big_ids = torch.nonzero(big_cluster_mask, as_tuple=False).flatten()
      if big_ids.numel() > 0:
        # small_centers = cluster_feat[need_radius]
        # big_centers = cluster_feat[big_ids]
        small_centers = cluster_pos[need_radius]
        big_centers = cluster_pos[big_ids]
        if small_centers.numel() > 0 and big_centers.numel() > 0:
          knn = pytorch3d.ops.knn_points(
              small_centers.unsqueeze(0),
              big_centers.unsqueeze(0),
              K=1,
              return_nn=False,
              return_sorted=False,
          )
          min_dist = knn.dists.squeeze(0).squeeze(-1).sqrt()
          min_idx = knn.idx.squeeze(0).squeeze(-1)
          valid = min_idx >= 0
          within = valid & (min_dist <= radius)
          if within.any():
            reassignment[need_radius[within]] = big_ids[min_idx[within]]

    reassignment_mask = reassignment >= 0
    if not reassignment_mask.any():
      return cluster_id, mask

    cluster_map = torch.arange(num_clusters, device=device)
    cluster_map[reassignment_mask] = reassignment[reassignment_mask]

    remapped_cluster_id = cluster_map[cluster_id]
    updated_peak_f = cluster_feat[remapped_cluster_id].type_as(self.peak_f)
    merged_nodes = reassignment_mask[cluster_id]
    updated_mask = mask | merged_nodes

    _, final_cluster_id = torch.unique(remapped_cluster_id,
                                       sorted=True,
                                       return_inverse=True)
    self.peak_f = updated_peak_f
    print(
        f"Number of clusters after fragment merging: {final_cluster_id.max().item() + 1}"
    )
    return final_cluster_id, updated_mask

  def run(
      self,
      iter_max,
      shifted_tol,
      fragment_min_cluster_size=10,
      fragment_search_radius=None,
  ):
    """
    Run the mean shift algorithm and graph merging
    """
    with torch.no_grad():
      self.logger.info("Running the mean shift algorithm and graph merging")
      for i in range(iter_max):
        shifted, shifted_dist = self.step()
        self.peak_f = shifted
        if shifted_dist.max() < shifted_tol:
          self.logger.info(f"Mean shift algorithm converged at iteration {i}")
          break
      cluster_id, mask = self.merge()
      cluster_id, mask = self.merge_fragments(cluster_id, mask,
                                              fragment_min_cluster_size,
                                              fragment_search_radius)

      return cluster_id, mask, self.peak_f.float()
