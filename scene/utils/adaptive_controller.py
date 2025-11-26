import os
import torch
from torch import nn
import torch_scatter
import numpy as np
from plyfile import PlyData, PlyElement

from functools import reduce
from einops import rearrange, repeat
from pytorch3d.ops import knn_points

from .. import GaussianModel


def _filter_new_anchor_grid_coords(selected_grid_coords_unique,
                                   grid_coords,
                                   duplicate_eps=1e-4):
  """Return mask for unseen coords and their nearest anchor indices."""
  if selected_grid_coords_unique.numel() == 0:
    keep_mask = torch.zeros_like(selected_grid_coords_unique[:, 0],
                                 dtype=torch.bool)
    return keep_mask, None

  if grid_coords.numel() == 0:
    keep_mask = torch.ones_like(selected_grid_coords_unique[:, 0],
                                dtype=torch.bool)
    return keep_mask, None

  query = selected_grid_coords_unique.unsqueeze(0).float()
  reference = grid_coords.unsqueeze(0).float()
  knn = knn_points(query, reference, K=1, return_nn=False)
  dists = knn.dists.squeeze(0).squeeze(-1)
  nn_indices = knn.idx.squeeze(0).squeeze(-1)
  keep_mask = dists > duplicate_eps
  return keep_mask, nn_indices


class AdaptiveController_scaffoldgs_style:

  def __init__(self, logger, gaussian_model: GaussianModel,
               grad_threshold: float, opacity_threshold: float):
    self.logger = logger
    self.model = gaussian_model
    self.grad_threshold = grad_threshold
    self.opacity_threshold = opacity_threshold
    self.init_children_num = self.model.init_child_num
    self.max_children_num = 1 * self.model.init_child_num
    self.extent = None
    self.max_screen_size = None

    self.stds_scale = 3.

  def gaussian_growing(self):
    # scaffold-GS does not need to grow Gaussian points
    pass

  def anchor_growing(self):
    self.logger.info("Growing Anchors.")
    child_grads = self.model.child_offset_gradient_accum / self.model.child_offset_denom  # (N, 1)
    child_grads = child_grads.view(-1)
    anchor_large_mask = self.model._anchor_radii > self.model.voxel_size * self.model.update_init_factor

    init_length = self.model._offset.shape[0]
    for i in range(self.model.update_depth):
      # update threshold
      cur_threshold = self.grad_threshold * (
          self.model.update_hierachy_factor // 2)**i
      # mask from grad threshold
      candidate_mask = (child_grads > cur_threshold)
      candidate_mask = candidate_mask & self.child_denom_mask

      # random pick
      rand_mask = torch.rand_like(candidate_mask.float(),
                                  device=candidate_mask.device) > (0.5**(i + 1))
      candidate_mask = candidate_mask & rand_mask

      # if we had append the new anchor in last loop, we should expend the cdandidate_mask
      length_inc = self.model._offset.shape[0] - init_length
      if length_inc == 0:
        if i > 0:
          continue
      else:
        candidate_mask = torch.cat(
            [
                candidate_mask,
                torch.zeros(
                    length_inc, dtype=torch.bool, device=candidate_mask.device)
            ],
            dim=0,
        )  # (N) -> (Ni)
      all_xyz = self.model.get_xyz  # (Ni, 3)

      # compute thr current voxel size
      size_factor = self.model.update_init_factor // (
          self.model.update_hierachy_factor**i)
      cur_size = self.model.voxel_size * size_factor

      # compute the anchor's grid coordinate
      grid_coords = torch.round(self.model._anchor_xyz /
                                cur_size).int()  # (Ai, 3)

      selected_xyz = all_xyz[candidate_mask]  # (ni, 3)

      # random sample the new position
      stds = self.model.scaling_activation(
          self.model._scaling)[candidate_mask].contiguous(
          ) * self.stds_scale  # (ni, 3)
      samples = torch.normal(mean=0, std=stds)  # (ni, 3)
      rots = self.model.quat2mat3(
          self.model.rotation_activation(
              self.model._rotation[candidate_mask].contiguous()))
      samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)  # (niur, 3)
      selected_xyz = selected_xyz + samples  # (ni, 3)

      selected_grid_coords = torch.round(selected_xyz /
                                         cur_size).int()  # (ni, 3)

      selected_grid_coords_unique, inverse_indices = torch.unique(
          selected_grid_coords, return_inverse=True,
          dim=0)  # (ni, 3) -> (niu, 3), index(ni)

      keep_mask, nearest_anchor_idx = _filter_new_anchor_grid_coords(
          selected_grid_coords_unique, grid_coords)
      candidate_anchor = selected_grid_coords_unique[
          keep_mask] * cur_size  # (niur, 3)

      if candidate_anchor.shape[0] == 0:
        continue

      if nearest_anchor_idx is not None:
        nearest_anchor_idx = nearest_anchor_idx[keep_mask]

      big_scale = torch_scatter.scatter_mean(
          big_scale[candidate_mask],
          inverse_indices.unsqueeze(1).expand(-1, big_scale.shape[1]),
          dim=0)[keep_mask]

      self.logger.info(
          f"[{i}/{self.model.update_depth}] Growing {candidate_anchor.shape[0]} anchors."
      )

      # anchor growing now!
      new_anchors_xyz = candidate_anchor  # (niur, 3) -> (a',3)
      new_anchors_radii = torch.ones(
          new_anchors_xyz.shape[0], 1,
          device=new_anchors_xyz.device) * cur_size * 1.0  # (a',1)
      if nearest_anchor_idx is not None:
        new_anchors_feature = self.model._feature[nearest_anchor_idx.to(
            self.model._feature.device)]
      else:
        fallback_feature = self.model._feature[
            self.model._point2anchor][candidate_mask]
        new_anchors_feature = torch_scatter.scatter_mean(
            fallback_feature,
            inverse_indices.unsqueeze(1).expand(-1, fallback_feature.shape[1]),
            dim=0)[keep_mask]  # (a', f)
      da = {
          "a_xyz": new_anchors_xyz,
          "a_feature": new_anchors_feature,
      }

      new_child_offset = repeat(self.model.offset_inverse_activation(
          torch.zeros_like(new_anchors_xyz)),
                                "n d -> (n c) d",
                                c=self.init_children_num)
      # stds = stds[remove_duplicates] * 0.1  # (niur, 3)
      # stds = repeat(stds, "n d -> (n c) d",
      #               c=self.init_children_num)  # (niur*c, 3)
      # rots = rots[remove_duplicates]  # (niur, 3, 3)
      # rots = repeat(rots, "n d l -> (n c) d l",
      #               c=self.init_children_num)  # (niur*c, 3, 3)
      # samples = torch.normal(mean=0, std=stds)  # (niur*c, 3)
      # samples = torch.bmm(rots,
      #                     samples.unsqueeze(-1)).squeeze(-1)  # (niur*c, 3)
      # samples = samples / repeat(new_anchors_radii,
      #                            "n r -> (n c) r",
      #                            c=self.init_children_num)  # (niur*c, 3)
      # new_child_offset = new_child_offset + samples  # (niur*c, 3)

      # generate new anchor's children
      new_rots = torch.zeros((new_anchors_xyz.shape[0], 4),
                             device=new_child_offset.device)
      new_rots[:, 0] = 1
      dc = {
          "c_offset":
              new_child_offset,
          "f_dc":
              repeat(torch_scatter.scatter_mean(
                  self.model._features_dc[candidate_mask],
                  inverse_indices.unsqueeze(1).expand(
                      -1, self.model._features_dc.shape[1]),
                  dim=0)[keep_mask],
                     "n d f -> (n c) d f",
                     c=self.init_children_num),
          "f_rest":
              repeat(torch_scatter.scatter_mean(
                  self.model._features_rest[candidate_mask],
                  inverse_indices.unsqueeze(1).expand(
                      -1, self.model._features_rest.shape[1]),
                  dim=0)[keep_mask],
                     "n d f -> (n c) d f",
                     c=self.init_children_num),
          "c_opacity":
              repeat(self.model.inverse_opacity_activation(
                  0.01 * (0.6 + 0.4 / self.init_children_num) *
                  torch.ones(new_anchors_xyz.shape[0],
                             1,
                             device=new_anchors_xyz.device)),
                     "n o -> (n c) o",
                     c=self.init_children_num),
          # "c_opacity":
          #     repeat(self.model._opacity[selected_children],
          #            "n o -> (n c) o",
          #            c=self.init_children_num),
          "c_scaling":
              repeat(self.model.scaling_inverse_activation_only(
                  0.8 * torch.ones_like(new_anchors_xyz)),
                     "n s -> (n c) s",
                     c=self.init_children_num),
          "c_rotation":
              repeat(new_rots, "n q -> (n c) q", c=self.init_children_num),
      }
      opt_tensors = self.model.OptModify.cat(self.model.anchor_optimizer, da)
      self.model.update_params(opt_tensors)

      new_point2anchor = torch.arange(self.model._anchor_radii.shape[0],
                                      self.model._anchor_radii.shape[0] +
                                      new_anchors_radii.shape[0],
                                      device=new_anchors_xyz.device)  # (a')

      self.model._point2anchor = torch.cat([
          self.model._point2anchor,
          repeat(new_point2anchor, 'n -> (n c)', c=self.init_children_num)
      ])  # (A+a')

      self.model._anchor_radii = torch.cat(
          [self.model._anchor_radii,
           new_anchors_radii.view(-1, 1)])  # (A+a', 1)

      opt_tensors = self.model.OptModify.cat(self.model.child_optimizer, dc)
      self.model.update_params(opt_tensors)

      self.model.child_offset_gradient_accum = torch.cat([
          self.model.child_offset_gradient_accum,
          torch.zeros((self.init_children_num * new_anchors_xyz.shape[0], 1),
                      device=new_anchors_xyz.device)
      ])
      self.model.child_offset_denom = torch.cat([
          self.model.child_offset_denom,
          torch.zeros((self.init_children_num * new_anchors_xyz.shape[0], 1),
                      device=new_anchors_xyz.device)
      ])
      self.model.max_radii2D = torch.cat([
          self.model.max_radii2D,
          torch.zeros(self.init_children_num * new_anchors_xyz.shape[0],
                      device=new_anchors_xyz.device)
      ])
    #   self.denom_mask = torch.cat([
    #       self.denom_mask,
    #       torch.ones((self.init_children_num * new_anchors_xyz.shape[0]),
    #                  device=new_anchors_xyz.device,
    #                  dtype=torch.bool)
    #   ])

  def gaussian_prune(self, extent, max_screen_size, mask=None):
    if mask is None:
      # mask = self.model.get_opacity.view(-1) < self.opacity_threshold
      # mask: child 2 anchor
      anchor_mean_opacity = self.model.scatter_mean_to_anchor(
          self.model.get_opacity).view(-1)
      mask = anchor_mean_opacity < self.opacity_threshold
      mask = mask[self.model._point2anchor]
      # if max_screen_size is not None:
      #   # mask = mask | (self.model.max_radii2D > max_screen_size)
      #   mask = mask | (self.model.get_scaling.max(dim=-1).values > 0.1 * extent)
    mask = ~mask
    self.logger.debug(f"Total Gaussian points: {mask.shape[0]}.")
    self.logger.info(f"Pruning {mask.shape[0]-mask.sum()} Gaussian points.")

    # mask = ~void_gaussian_mask
    opt_tensors = self.model.OptModify.remove(self.model.child_optimizer, mask)
    self.model.update_params(opt_tensors)

    self.model._point2anchor = self.model._point2anchor[mask]
    self.model.child_offset_gradient_accum = self.model.child_offset_gradient_accum[
        mask]
    self.model.child_offset_denom = self.model.child_offset_denom[mask]
    self.model.max_radii2D = self.model.max_radii2D[mask]

  def anchor_prune(self):
    # self.logger.debug("Anchor shape: {}".format(self.model._anchor_radii.shape))
    count = self.model._point2anchor.bincount(
        minlength=self.model._anchor_radii.shape[0])  # (A)
    mask = count == 0  # (A)

    vaild_mask = ~mask
    self.logger.info(f"Pruning {mask.sum()} anchors.")

    modify_anchor_index = torch.arange(self.model._anchor_radii.shape[0],
                                       dtype=self.model._point2anchor.dtype,
                                       device=mask.device)  # (A)

    opt_tensors = self.model.OptModify.remove(self.model.anchor_optimizer,
                                              vaild_mask)
    self.model.update_params(opt_tensors)
    self.model._anchor_radii = self.model._anchor_radii[vaild_mask]

    # if mask = [False, True, True, False, True], which means 5 anchors, 1, 2, 4 are pruned (2 anchors left)
    #   0,3 are left, so modify_anchor_index must is [0, x, x, 1, x]
    # torch.cumsum(mask,0) can get the offset of the index after pruned:
    #   modify_anchor_index = [0, 1, 2, 3, 4] - [0, 1, 2, 2, 3] = [0, 0, 0, 1, 1]
    modify_anchor_index = modify_anchor_index - torch.cumsum(
        mask, 0, dtype=modify_anchor_index.dtype)
    self.model._point2anchor = modify_anchor_index[self.model._point2anchor]

  def growing(self):
    self.logger.info("Growing Anchors and Gaussian points.")
    self.child_denom_mask = (self.model.child_offset_denom > 10).int()
    self.anchor_denom_mask = self.model.scatter_max_to_anchor(
        self.child_denom_mask).view(-1).bool()
    # self.child_denom_mask = self.anchor_denom_mask[self.model._point2anchor]
    self.child_denom_mask = self.child_denom_mask.view(-1).bool()
    self.gaussian_growing()
    torch.cuda.empty_cache()
    self.anchor_growing()
    # self.model.child_offset_gradient_accum[self.child_denom_mask] = 0
    # self.model.child_offset_denom[self.child_denom_mask] = 0
    # self.model.max_radii2D = torch.zeros_like(self.model.max_radii2D)
    del self.child_denom_mask, self.anchor_denom_mask
    torch.cuda.empty_cache()

  def prune(self):
    self.logger.info("Pruning Anchors and Gaussian points.")
    self.child_denom_mask = (self.model.child_offset_denom > 20).int()
    self.anchor_denom_mask = self.model.scatter_max_to_anchor(
        self.child_denom_mask).view(-1).bool()
    # self.child_denom_mask = self.anchor_denom_mask[self.model._point2anchor]
    self.child_denom_mask = self.child_denom_mask.view(-1).bool()
    self.model.child_offset_gradient_accum[self.child_denom_mask] = 0
    self.model.child_offset_denom[self.child_denom_mask] = 0
    self.model.max_radii2D = torch.zeros_like(self.model.max_radii2D)
    self.gaussian_prune(self.extent, self.max_screen_size)
    torch.cuda.empty_cache()
    self.anchor_prune()
    del self.child_denom_mask, self.anchor_denom_mask
    torch.cuda.empty_cache()

  def control(self):
    self.logger.info(
        f"Control info: grad_threshold {self.grad_threshold}, opacity_threshold {self.opacity_threshold}."
    )
    self.growing()
    self.prune()
    # self.model.clean_densification_states()


class AdaptiveController_scaffoldgs_style_free_scale:

  def __init__(self, logger, gaussian_model: GaussianModel,
               grad_threshold: float, opacity_threshold: float):
    self.logger = logger
    self.model = gaussian_model
    self.grad_threshold = grad_threshold
    self.opacity_threshold = opacity_threshold
    self.init_children_num = self.model.init_child_num
    self.max_children_num = 1 * self.model.init_child_num
    self.extent = None
    self.max_screen_size = None

    self.stds_scale = 1.

  def gaussian_growing(self):
    # scaffold-GS does not need to grow Gaussian points
    pass

  def anchor_growing(self):
    self.logger.info("Growing Anchors.")
    child_grads = self.model.child_offset_gradient_accum / self.model.child_offset_denom  # (N, 1)
    child_dis = 1e-4 * torch.norm(self.model._offset, p=2, dim=-1,
                                  keepdim=True)  # (N, 1)
    child_dis += 5e-5 * torch.relu(
        torch.max(self.model.get_scaling, dim=-1, keepdim=True).values -
        self.model._anchor_radii[self.model._point2anchor])  # (N, 1)
    self.logger.info(
        f"{(child_grads>self.grad_threshold).sum()} Gaussian points are larger than {self.grad_threshold}."
    )
    child_grads += child_dis
    anchor_grads = self.model.scatter_mean_to_anchor(child_grads).view(
        -1)  # (A)
    child_grads = child_grads.view(-1)
    anchor_large_mask = self.model._anchor_radii > self.model.voxel_size * self.model.update_init_factor
    child_large_mask = anchor_large_mask[
        self.model._point2anchor].contiguous().view(-1)

    big_gs = self.model.get_scaling.max(dim=-1).values > 0.1 * self.extent
    if self.max_screen_size is not None:
      big_gs = big_gs | (self.model.max_radii2D > self.max_screen_size)

    init_length = self.model._offset.shape[0]
    for i in range(self.model.update_depth):
      # update threshold
      cur_threshold = self.grad_threshold * (
          self.model.update_hierachy_factor // 2)**i

      big_scale = 1.0 - 0.0 * big_gs.float().view(-1, 1)

      # mask from grad threshold
      candidate_mask = (child_grads > cur_threshold)
      # rand_mask = torch.rand_like(big_gs.float(), device=big_gs.device) > 0.95
      # print(f"rand_mask {rand_mask.shape} {big_gs.shape} {big_scale.shape}")
      #   candidate_mask = candidate_mask | (big_gs & rand_mask)
      candidate_mask = candidate_mask & self.child_denom_mask

      if False and i == 0:
        candidate_mask = candidate_mask & child_large_mask & (torch.rand_like(
            candidate_mask.float(), device=candidate_mask.device) > 0.9)
      # random pick
      rand_mask = torch.rand_like(candidate_mask.float(),
                                  device=candidate_mask.device) > (0.5**(i + 1))
      candidate_mask = candidate_mask & rand_mask

      # if we had append the new anchor in last loop, we should expend the cdandidate_mask
      length_inc = self.model._offset.shape[0] - init_length
      if length_inc == 0:
        if i > 0:
          continue
      else:
        candidate_mask = torch.cat(
            [
                candidate_mask,
                torch.zeros(
                    length_inc, dtype=torch.bool, device=candidate_mask.device)
            ],
            dim=0,
        )  # (N) -> (Ni)
        big_scale = torch.cat(
            [
                big_scale,
                torch.ones(length_inc,
                           1,
                           dtype=big_scale.dtype,
                           device=big_scale.device)
            ],
            dim=0,
        )
      all_xyz = self.model.get_xyz  # (Ni, 3)

      # compute thr current voxel size
      size_factor = self.model.update_init_factor // (
          self.model.update_hierachy_factor**i)
      cur_size = self.model.voxel_size * size_factor

      # compute the anchor's grid coordinate
      grid_coords = torch.round(self.model._anchor_xyz /
                                cur_size).int()  # (Ai, 3)

      selected_xyz = all_xyz[candidate_mask]  # (ni, 3)

      # random sample the new position

      if False:
        stds_scale = self.model.update_depth - i + 0.0
      try:
        stds = self.model.scaling_activation(
            self.model._scaling)[candidate_mask].contiguous(
            ) * self.stds_scale  # (ni, 3)
        # replace nan to 0
        # stds = torch.where(stds.isnan(), torch.zeros_like(stds), stds)
        samples = torch.normal(mean=0, std=stds)  # (ni, 3)
      except Exception as e:
        print(
            f"Error: {e}, {self.model._scaling.shape} {self.model._scaling[candidate_mask].shape}, {(stds<0).sum()}, nan check {stds.isnan().sum()}, {stds.isinf().sum()}"
        )
      rots = self.model.quat2mat3(
          self.model.rotation_activation(
              self.model._rotation[candidate_mask].contiguous()))
      samples = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)  # (niur, 3)
      selected_xyz = selected_xyz + samples  # (ni, 3)

      selected_grid_coords = torch.round(selected_xyz /
                                         cur_size).int()  # (ni, 3)

      selected_grid_coords_unique, inverse_indices = torch.unique(
          selected_grid_coords, return_inverse=True,
          dim=0)  # (ni, 3) -> (niu, 3), index(ni)

      keep_mask, nearest_anchor_idx = _filter_new_anchor_grid_coords(
          selected_grid_coords_unique, grid_coords)
      candidate_anchor = selected_grid_coords_unique[
          keep_mask] * cur_size  # (niur, 3)

      if candidate_anchor.shape[0] == 0:
        continue

      if nearest_anchor_idx is not None:
        nearest_anchor_idx = nearest_anchor_idx[keep_mask]

      big_scale = torch_scatter.scatter_mean(
          big_scale[candidate_mask],
          inverse_indices.unsqueeze(1).expand(-1, big_scale.shape[1]),
          dim=0)[keep_mask]

      self.logger.info(
          f"[{i}/{self.model.update_depth}] Growing {candidate_anchor.shape[0]} anchors."
      )

      # anchor growing now!
      new_anchors_xyz = candidate_anchor  # (niur, 3) -> (a',3)
      new_anchors_radii = torch.ones(
          new_anchors_xyz.shape[0], 1,
          device=new_anchors_xyz.device) * cur_size * 1.0  # (a',1)
      if nearest_anchor_idx is not None:
        new_anchors_feature = self.model._feature[nearest_anchor_idx.to(
            self.model._feature.device)]
      else:
        fallback_feature = self.model._feature[
            self.model._point2anchor][candidate_mask]
        new_anchors_feature = torch_scatter.scatter_mean(
            fallback_feature,
            inverse_indices.unsqueeze(1).expand(-1, fallback_feature.shape[1]),
            dim=0)[keep_mask]  # (a', f)
      da = {
          "a_xyz": new_anchors_xyz,
          "a_feature": new_anchors_feature,
      }

      new_child_offset = repeat(self.model.offset_inverse_activation(
          torch.zeros_like(new_anchors_xyz)),
                                "n d -> (n c) d",
                                c=self.init_children_num)
      # stds = stds[remove_duplicates] * 0.1  # (niur, 3)
      # stds = repeat(stds, "n d -> (n c) d",
      #               c=self.init_children_num)  # (niur*c, 3)
      # rots = rots[remove_duplicates]  # (niur, 3, 3)
      # rots = repeat(rots, "n d l -> (n c) d l",
      #               c=self.init_children_num)  # (niur*c, 3, 3)
      # samples = torch.normal(mean=0, std=stds)  # (niur*c, 3)
      # samples = torch.bmm(rots,
      #                     samples.unsqueeze(-1)).squeeze(-1)  # (niur*c, 3)
      # samples = samples / repeat(new_anchors_radii,
      #                            "n r -> (n c) r",
      #                            c=self.init_children_num)  # (niur*c, 3)
      # new_child_offset = new_child_offset + samples  # (niur*c, 3)

      # generate new anchor's children
      new_rots = torch.zeros((new_anchors_xyz.shape[0], 4),
                             device=new_child_offset.device)
      new_rots[:, 0] = 1
      dc = {
          "c_offset":
              new_child_offset,
          "f_dc":
              repeat(torch_scatter.scatter_mean(
                  self.model._features_dc[candidate_mask],
                  inverse_indices.unsqueeze(1).expand(
                      -1, self.model._features_dc.shape[1]),
                  dim=0)[keep_mask],
                     "n d f -> (n c) d f",
                     c=self.init_children_num),
          "f_rest":
              repeat(torch_scatter.scatter_mean(
                  self.model._features_rest[candidate_mask],
                  inverse_indices.unsqueeze(1).expand(
                      -1, self.model._features_rest.shape[1]),
                  dim=0)[keep_mask],
                     "n d f -> (n c) d f",
                     c=self.init_children_num),
          #   "c_opacity":
          #       repeat(self.model.inverse_opacity_activation(
          #           0.01 * (0.6 + 0.4 / self.init_children_num) *
          #           torch.ones(new_anchors_xyz.shape[0],
          #                      1,
          #                      device=new_anchors_xyz.device)),
          #              "n o -> (n c) o",
          #              c=self.init_children_num),
          "c_opacity":
              repeat(self.model.inverse_opacity_activation(0.01 * torch.ones(
                  new_anchors_xyz.shape[0], 1, device=new_anchors_xyz.device)),
                     "n o -> (n c) o",
                     c=self.init_children_num),
          # "c_opacity":
          #     repeat(self.model._opacity[selected_children],
          #            "n o -> (n c) o",
          #            c=self.init_children_num),
          #   "c_scaling":
          #       repeat(self.model.scaling_inverse_activation_only(
          #           (0.8 * big_scale) * torch.ones_like(new_anchors_xyz)),
          #              "n s -> (n c) s",
          #              c=self.init_children_num),
          "c_scaling":
              repeat(self.model.scaling_inverse_activation(
                  (0.8 * big_scale) * new_anchors_radii.repeat(1, 3)),
                     "n s -> (n c) s",
                     c=self.init_children_num),
          "c_rotation":
              repeat(new_rots, "n q -> (n c) q", c=self.init_children_num),
      }
      opt_tensors = self.model.OptModify.cat(self.model.anchor_optimizer, da)
      self.model.update_params(opt_tensors)

      new_point2anchor = torch.arange(self.model._anchor_radii.shape[0],
                                      self.model._anchor_radii.shape[0] +
                                      new_anchors_radii.shape[0],
                                      device=new_anchors_xyz.device)  # (a')

      self.model._point2anchor = torch.cat([
          self.model._point2anchor,
          repeat(new_point2anchor, 'n -> (n c)', c=self.init_children_num)
      ])  # (A+a')

      self.model._anchor_radii = torch.cat(
          [self.model._anchor_radii,
           new_anchors_radii.view(-1, 1)])  # (A+a', 1)

      opt_tensors = self.model.OptModify.cat(self.model.child_optimizer, dc)
      self.model.update_params(opt_tensors)

      self.model.child_offset_gradient_accum = torch.cat([
          self.model.child_offset_gradient_accum,
          torch.zeros((self.init_children_num * new_anchors_xyz.shape[0], 1),
                      device=new_anchors_xyz.device)
      ])
      self.model.child_offset_denom = torch.cat([
          self.model.child_offset_denom,
          torch.zeros((self.init_children_num * new_anchors_xyz.shape[0], 1),
                      device=new_anchors_xyz.device)
      ])
      self.model.max_radii2D = torch.cat([
          self.model.max_radii2D,
          torch.zeros(self.init_children_num * new_anchors_xyz.shape[0],
                      device=new_anchors_xyz.device)
      ])
    #   self.denom_mask = torch.cat([
    #       self.denom_mask,
    #       torch.ones((self.init_children_num * new_anchors_xyz.shape[0]),
    #                  device=new_anchors_xyz.device,
    #                  dtype=torch.bool)
    #   ])

  def gaussian_prune(self, extent, max_screen_size, mask=None):
    if mask is None:
      # mask = self.model.get_opacity.view(-1) < self.opacity_threshold
      # mask: child 2 anchor
      anchor_mean_opacity = self.model.scatter_mean_to_anchor(
          self.model.get_opacity).view(-1)
      mask = anchor_mean_opacity < self.opacity_threshold
      mask = mask[self.model._point2anchor]

      c_nan = self.model.get_scaling.isnan()
      if c_nan.any():
        c_nan = c_nan.sum(dim=-1) > 0
        c_sum_m = self.model.scatter_add_to_anchor(c_nan.float().view(
            -1, 1)).view(-1)
        mask = mask | ((c_sum_m > 0.1)[self.model._point2anchor])

      if True and max_screen_size is not None:
        c_large_m = (self.model.get_scaling.max(dim=-1).values > 0.1 * extent)
        c_large_m = c_large_m | (self.model.max_radii2D > max_screen_size)

        # scatter to anchor
        a_mean_m = self.model.scatter_mean_to_anchor(c_large_m.float().view(
            -1, 1)).view(-1)
        mask = mask | ((a_mean_m > 0.3)[self.model._point2anchor])

    mask = ~mask
    self.logger.debug(f"Total Gaussian points: {mask.shape[0]}.")
    self.logger.info(f"Pruning {mask.shape[0]-mask.sum()} Gaussian points.")

    # mask = ~void_gaussian_mask
    opt_tensors = self.model.OptModify.remove(self.model.child_optimizer, mask)
    self.model.update_params(opt_tensors)

    self.model._point2anchor = self.model._point2anchor[mask]
    self.model.child_offset_gradient_accum = self.model.child_offset_gradient_accum[
        mask]
    self.model.child_offset_denom = self.model.child_offset_denom[mask]
    self.model.max_radii2D = self.model.max_radii2D[mask]

  def anchor_prune(self):
    # self.logger.debug("Anchor shape: {}".format(self.model._anchor_radii.shape))
    count = self.model._point2anchor.bincount(
        minlength=self.model._anchor_radii.shape[0])  # (A)
    mask = count == 0  # (A)

    vaild_mask = ~mask
    self.logger.info(f"Pruning {mask.sum()} anchors.")

    modify_anchor_index = torch.arange(self.model._anchor_radii.shape[0],
                                       dtype=self.model._point2anchor.dtype,
                                       device=mask.device)  # (A)

    opt_tensors = self.model.OptModify.remove(self.model.anchor_optimizer,
                                              vaild_mask)
    self.model.update_params(opt_tensors)
    self.model._anchor_radii = self.model._anchor_radii[vaild_mask]

    # if mask = [False, True, True, False, True], which means 5 anchors, 1, 2, 4 are pruned (2 anchors left)
    #   0,3 are left, so modify_anchor_index must is [0, x, x, 1, x]
    # torch.cumsum(mask,0) can get the offset of the index after pruned:
    #   modify_anchor_index = [0, 1, 2, 3, 4] - [0, 1, 2, 2, 3] = [0, 0, 0, 1, 1]
    modify_anchor_index = modify_anchor_index - torch.cumsum(
        mask, 0, dtype=modify_anchor_index.dtype)
    self.model._point2anchor = modify_anchor_index[self.model._point2anchor]

  def growing(self):
    self.logger.info("Growing Anchors and Gaussian points.")
    self.child_denom_mask = (self.model.child_offset_denom > 10).int()
    self.anchor_denom_mask = self.model.scatter_max_to_anchor(
        self.child_denom_mask).view(-1).bool()
    # self.child_denom_mask = self.anchor_denom_mask[self.model._point2anchor]
    self.child_denom_mask = self.child_denom_mask.view(-1).bool()
    self.gaussian_growing()
    torch.cuda.empty_cache()
    self.anchor_growing()
    # self.model.child_offset_gradient_accum[self.child_denom_mask] = 0
    # self.model.child_offset_denom[self.child_denom_mask] = 0
    # self.model.max_radii2D = torch.zeros_like(self.model.max_radii2D)
    del self.child_denom_mask, self.anchor_denom_mask
    torch.cuda.empty_cache()

  def prune(self):
    self.logger.info("Pruning Anchors and Gaussian points.")
    self.child_denom_mask = (self.model.child_offset_denom > 10).int()
    self.anchor_denom_mask = self.model.scatter_max_to_anchor(
        self.child_denom_mask).view(-1).bool()
    # self.child_denom_mask = self.anchor_denom_mask[self.model._point2anchor]
    self.child_denom_mask = self.child_denom_mask.view(-1).bool()
    self.model.child_offset_gradient_accum[self.child_denom_mask] = 0
    self.model.child_offset_denom[self.child_denom_mask] = 0
    self.model.max_radii2D = torch.zeros_like(self.model.max_radii2D)
    self.gaussian_prune(self.extent, self.max_screen_size)
    torch.cuda.empty_cache()
    self.anchor_prune()
    del self.child_denom_mask, self.anchor_denom_mask
    torch.cuda.empty_cache()

  def control(self):
    self.logger.info(
        f"Control info: grad_threshold {self.grad_threshold}, opacity_threshold {self.opacity_threshold}."
    )
    self.growing()
    self.prune()


# AdaptiveController = AdaptiveController_3dgs_style
AdaptiveController = AdaptiveController_scaffoldgs_style
AdaptiveController_free_scale = AdaptiveController_scaffoldgs_style_free_scale
