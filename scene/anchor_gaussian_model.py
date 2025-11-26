#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import time
import torch
import torch_scatter
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.math_utils import *

from einops import rearrange, repeat
import ipdb

import pytorch3d.ops

from tqdm.auto import tqdm


class GaussianModel:

  def setup_functions(self):

    def build_covariance_from_scaling_rotation(scaling, scaling_modifier,
                                               rotation):
      # Rotate @ scale
      L = build_scaling_rotation(scaling_modifier * scaling, rotation)
      # Covariance is L @ L^T = R @ S @ S^T @ R^T
      actual_covariance = L @ L.transpose(1, 2)
      # Transform symmetric matrix to lower triangular: (N, 3, 3) -> (N, 6)
      symm = strip_symmetric(actual_covariance)
      return symm

    if self.use_free_scale:
      self.logger.info("Using free scaling")
      self.scaling_activation = torch.exp
      self.scaling_inverse_activation = torch.log
      self.scaling_inverse_activation_only = torch.log
    else:
      self.logger.info("Using anchor radii for scaling")
      self.scaling_activation = lambda s: torch.sigmoid(s) * self._anchor_radii[
          self._point2anchor].contiguous()
      self.scaling_inverse_activation = lambda s: inverse_sigmoid(
          torch.clamp(s / self._anchor_radii[self._point2anchor].contiguous(),
                      1e-4, 0.99))
      self.scaling_inverse_activation_only = lambda s: inverse_sigmoid(s)

    self.covariance_activation = build_covariance_from_scaling_rotation

    # Opacity is a value between 0 and 1
    self.opacity_activation = torch.sigmoid
    # Inverse value between 0 and 1 to opacity
    self.inverse_opacity_activation = inverse_sigmoid

    self.rotation_activation = torch.nn.functional.normalize

    self.segment_activation = lambda f: torch.nn.functional.normalize(f, dim=-1)
    self.segment_inverse_activation = lambda f: f
    # self.segment_activation = lambda f: torch.nn.functional.sigmoid(f) * 2. - 1
    # self.segment_inverse_activation = lambda f: inverse_sigmoid(f * .5 + .5)

    # self.offset_activation = lambda o: torch.sigmoid(o) - 0.5
    # self.offset_inverse_activation = lambda o: inverse_sigmoid(o + 0.5)
    self.offset_activation = lambda o: o
    self.offset_inverse_activation = lambda o: o

    self.quat2mat3 = build_rotation

  def __init__(
      self,
      sh_degree: int,
      init_child_num: int = 5,
      voxel_size: float = 0.0,
      update_depth: int = 3,
      update_init_factor: int = 16,
      update_hierachy_factor: int = 4,
      use_free_scale=None,
      voxel_size_scale=None,
      logger=None,
      training_args=None,
  ):
    if logger is not None:
      self.logger = logger
    else:
      import logging
      # get a blackhole logger
      self.logger = logging.getLogger("blackhole")

    if use_free_scale is not None:
      self.use_free_scale = use_free_scale > 0.5
    else:
      self.logger.warning(
          "Input args does not have use_free_scale, set to False")
      self.use_free_scale = False
    self.logger.info(f"use free scale: {self.use_free_scale}")

    # scaffold-gs' voxel grid parameters ------------------------
    self.voxel_size = voxel_size
    self.update_depth = update_depth
    self.update_init_factor = update_init_factor
    self.update_hierachy_factor = update_hierachy_factor
    if voxel_size_scale is not None and voxel_size_scale > 0:
      self.logger.info(f"Using voxel size scale: {voxel_size_scale}")
      self.voxel_size_scale = voxel_size_scale
    else:
      self.voxel_size_scale = 1.0
    # -----------------------------------------------------------

    self.active_sh_degree = 0
    self.max_sh_degree = sh_degree
    # anchor points, 3 channels ---------------------------------
    self._anchor_xyz = torch.empty(0)  # (A, 3)
    self._anchor_radii = torch.empty(0)  # (A, 1)

    self._feature = torch.empty(0)  # (A, 3)
    # -----------------------------------------------------------

    # children offset, 3 channels -------------------------------
    self._offset = torch.empty(0)  # (N, 3)
    # -----------------------------------------------------------
    # index from child points to anchor, int --------------------
    self._point2anchor = torch.empty(0)  # (N), torch.int32
    # -----------------------------------------------------------

    # -----------------------------------------------------------
    # restore the SH color, 3 channels---------------------------
    # SH_0, equal to RGB texture
    self._features_dc = torch.empty(0)  # (N, 3, SH_0=1)
    # SH_1:max
    self._features_rest = torch.empty(0)  # (N, 3, SH_1:max)
    # -----------------------------------------------------------
    # scaling, rotation, opacity --------------------------------
    self._scaling = torch.empty(0)  # (N, 3)
    self._rotation = torch.empty(0)  # (N, 4)
    self._opacity = torch.empty(0)  # (N, 1)
    # -----------------------------------------------------------
    # -----------------------------------------------------------
    # save the max radii of the 2D Gaussians
    self.max_radii2D = torch.empty(0)  # (N)
    # 2D gradient accumulator, for densification and pruning ----
    self.child_offset_gradient_accum = torch.empty(0)  # (N, 1)
    self.child_offset_denom = torch.empty(0)  # (N, 1)
    # -----------------------------------------------------------
    self.child_optimizer = None
    self.anchor_optimizer = None
    self.percent_dense = 0
    self.spatial_lr_scale = 0
    self.init_child_num = init_child_num
    self.setup_functions()

    if True:
      from .utils.adaptive_controller import AdaptiveController, AdaptiveController_free_scale
      from .utils.optimizer_modifier import OptimizerModifier
      from .utils.loss_helper import LossHelper
      from .utils.ply_helper import PlyHelper
      from .utils.graph_helper import GraphHelper
      from .utils.lang_helper import LangHelper
      """
      In this paper, we use the Scaffold-GS style anchor-gaussian and ADC.
      However, it is quite dependent on the initial sparse point cloud.
      (If you want to initialize with a dense point cloud obtained from 3R or other works, it's okay!)

      Let your imagination run wild, you don't necessarily need the Scaffold-GS style ADC, there are many better ADCs!
      """
      if self.use_free_scale:
        self.logger.info("Using free scale")
        self.AdControl = AdaptiveController_free_scale(self.logger, self,
                                                       0.0002, 0.005)
      else:
        self.logger.info("Using localized scale")
        self.AdControl = AdaptiveController(self.logger, self, 0.0002, 0.005)

      self.OptModify = OptimizerModifier(self.logger, self)
      self.LossHelper = LossHelper(self.logger, self)
      self.PlyHelper = PlyHelper(self.logger, self)
      self.GraphHelper = GraphHelper(self.logger, self)
      self.LangHelper = LangHelper(self.logger, self)
      self._init_loss(training_args=training_args)

  def _init_loss(self, training_args):
    if self.use_free_scale:
      """
      We found that sigmoid-based localizing scaling constraints requires more accurate initial dense point clouds.
      Therefore, after careful analysis, we decided not to hard-constrain scaling, but instead use a soft loss similar to that used for offset.
      """
      self.logger.info("Using free scale loss")

      def _loss_limit_chlid_anchor_clustering():
        # return 0.0
        # get the max scale of the points (N, 3) -> (N, 1)
        max_scale = torch.max(self._scaling[self.update_filter],
                              dim=1,
                              keepdim=True)[0]
        max_scale = self.scaling_activation(max_scale)
        # get the distance between the children and anchor
        dis = torch.norm(self._offset[self.update_filter].contiguous(),
                         p=2,
                         dim=1,
                         keepdim=True)  # (N, 1)

        return (torch.relu(dis - 1).exp() - 1
               ).mean() + torch.relu(max_scale - self._anchor_radii[
                   self._point2anchor[self.update_filter]].contiguous()).mean()

    else:

      def _loss_limit_chlid_anchor_clustering():
        dis = torch.norm(self._offset, p=2, dim=1, keepdim=True)  # (N, 1)
        return (torch.relu(dis - 1).exp() - 1).mean()

    c_in_a_loss_weight = 0.5
    if training_args is not None:
      c_in_a_loss_weight = training_args.c_in_a_loss_weight
    self.LossHelper.append("children_in_anchor_loss", c_in_a_loss_weight,
                           _loss_limit_chlid_anchor_clustering)

    graph_loss_weight = 1e-2
    if training_args is not None:
      graph_loss_weight = training_args.graph_loss_weight
    self.LossHelper.append(
        "graph_loss", graph_loss_weight,
        lambda: self.GraphHelper.propagation_loss(self.update_filter))

  def get_loss(self):
    return self.LossHelper.get_loss()

  def capture(self):
    return (
        self.active_sh_degree,
        # -----------------
        self._anchor_xyz,
        self._anchor_radii,
        self._feature,
        self._offset,
        # self._anchor2point,
        self._point2anchor,
        # -----------------
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.child_offset_gradient_accum,
        self.child_offset_denom,
        self.child_optimizer.state_dict(),
        self.anchor_optimizer.state_dict(),
        self.spatial_lr_scale,
    )

  def restore(self, model_args, training_args):
    (
        self.active_sh_degree,
        # ------------------------------------------------------
        self._anchor_xyz,
        self._anchor_radii,
        self._feature,
        self._offset,
        # self._anchor2point,
        self._point2anchor,
        # ------------------------------------------------------
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        child_offset_gradient_accum,
        child_offset_denom,
        c_opt_dict,
        a_opt_dict,
        self.spatial_lr_scale) = model_args
    self.training_setup(training_args)
    self.child_offset_gradient_accum = child_offset_gradient_accum
    self.child_offset_denom = child_offset_denom
    self.child_optimizer.load_state_dict(c_opt_dict)
    self.anchor_optimizer.load_state_dict(a_opt_dict)

  @property
  def get_scaling(self):
    return self.scaling_activation(self._scaling)

  @property
  def get_rotation(self):
    return self.rotation_activation(self._rotation)

  @property
  def get_xyz(self):
    return (self._anchor_xyz[self._point2anchor] +
            self._anchor_radii[self._point2anchor] *
            self.offset_activation(self._offset)).contiguous()  # (N, 3)

  @property
  def get_child_anchor_xyz(self):
    return self._anchor_xyz[self._point2anchor].contiguous()

  @property
  def get_features(self):
    features_dc = self._features_dc
    features_rest = self._features_rest
    return torch.cat((features_dc, features_rest), dim=1)

  @property
  def get_semantics(self):
    # self.logger.debug("Device debug: {}".format(self._anchor_xyz.device))
    # self.logger.debug("Device debug: {}".format(self._point2anchor.device))
    # self.logger.debug("Device debug: {}".format(self._feature.device))
    return self.segment_activation(
        self._feature)[self._point2anchor].contiguous()

  @property
  def get_anchor_semantics(self):
    return self.segment_activation(self._feature)

  @property
  def get_anchor_index(self):
    features_color = torch.linspace(0,
                                    1,
                                    steps=self._anchor_xyz.shape[0],
                                    device=self._anchor_xyz.device)  # (A,)
    # shuffle the color
    features_color = features_color[torch.randperm(
        features_color.shape[0])][self._point2anchor]
    # color to RGB
    return value2HSV2RGB(features_color)  # (N,3)

  @property
  def get_opacity(self):
    return self.opacity_activation(self._opacity).contiguous()

  def get_covariance(self, scaling_modifier=1):
    return self.covariance_activation(self.get_scaling, scaling_modifier,
                                      self._rotation).view(-1, 6)

  @property
  def get_num_children_per_anchor(self):
    return self._point2anchor.bincount(minlength=self._anchor_xyz.shape[0])

  def scatter_add_to_anchor(self, tensor):
    aggregated = torch.zeros((self._anchor_xyz.shape[0], tensor.shape[1]),
                             dtype=tensor.dtype,
                             device="cuda")
    # return aggregated.scatter_add(0, self._point2anchor.long(), tensor)
    torch_scatter.scatter_add(tensor,
                              self._point2anchor.long(),
                              dim=0,
                              out=aggregated)
    return aggregated

  def scatter_max_to_anchor(self, tensor):
    aggregated = torch.zeros((self._anchor_xyz.shape[0], tensor.shape[1]),
                             dtype=tensor.dtype,
                             device="cuda")
    # print(aggregated.shape)
    # return aggregated.scatter_add(0, self._point2anchor.long(), tensor)
    torch_scatter.scatter_max(tensor,
                              self._point2anchor.long(),
                              dim=0,
                              out=aggregated)
    return aggregated

  def scatter_mean_to_anchor(self, tensor):
    aggregated = torch.zeros((self._anchor_xyz.shape[0], tensor.shape[1]),
                             dtype=tensor.dtype,
                             device="cuda")
    # return aggregated.scatter_mean(0, self._point2anchor.to(torch.int64),
    #                                tensor)
    torch_scatter.scatter_mean(tensor,
                               self._point2anchor.long(),
                               dim=0,
                               out=aggregated)
    return aggregated

  def gs2anchor_mask(self, point_visible_mask):
    return torch_scatter.scatter_add(point_visible_mask,
                                     self._point2anchor.long(),
                                     out=torch.zeros(
                                         self._anchor_xyz.shape[0],
                                         dtype=torch.bool,
                                         device=self._anchor_xyz.device))

  def oneupSHdegree(self):
    if self.active_sh_degree < self.max_sh_degree:
      self.active_sh_degree += 1

  def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    self.PlyHelper.create_from_pcd(pcd, spatial_lr_scale)

  def training_setup(self, training_args):
    self.GraphHelper.tau = training_args.graph_tau
    self.ts = training_args
    self.percent_dense = training_args.percent_dense
    self.child_offset_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1),
                                                   device="cuda")
    self.child_offset_denom = torch.zeros((self.get_xyz.shape[0], 1),
                                          device="cuda")
    lc = [{
        'params': [self._offset],
        'lr': training_args.offset_lr_init * self.spatial_lr_scale,
        "name": "c_offset"
    }, {
        'params': [self._features_dc],
        'lr': training_args.feature_lr,
        "name": "f_dc"
    }, {
        'params': [self._features_rest],
        'lr': training_args.feature_lr / 20.0,
        "name": "f_rest"
    }, {
        'params': [self._opacity],
        'lr': training_args.opacity_lr,
        "name": "c_opacity"
    }, {
        'params': [self._scaling],
        'lr': training_args.scaling_lr,
        "name": "c_scaling"
    }, {
        'params': [self._rotation],
        'lr': training_args.rotation_lr,
        "name": "c_rotation"
    }]
    for li in lc:
      self.logger.info("Learning rate for {} is {}".format(
          li["name"], li["lr"]))
    self.child_optimizer = torch.optim.Adam(lc, lr=0.0, eps=1e-15)

    la = [{
        'params': [self._anchor_xyz],
        'lr': training_args.position_lr_init * self.spatial_lr_scale,
        "name": "a_xyz"
    }, {
        'params': [self._feature],
        'lr': training_args.semantic_lr,
        "name": "a_feature"
    }]
    for li in la:
      self.logger.info("Learning rate for {} is {}".format(
          li["name"], li["lr"]))
    self.anchor_optimizer = torch.optim.Adam(la, lr=0.0, eps=1e-15)

    self.xyz_scheduler_args = get_expon_lr_func(
        lr_init=training_args.position_lr_init * self.spatial_lr_scale,
        lr_final=training_args.position_lr_final * self.spatial_lr_scale,
        lr_delay_mult=training_args.position_lr_delay_mult,
        max_steps=training_args.position_lr_max_steps)
    self.offset_scheduler_args = get_expon_lr_func(
        lr_init=training_args.offset_lr_init * self.spatial_lr_scale,
        lr_final=training_args.offset_lr_final * self.spatial_lr_scale,
        lr_delay_mult=training_args.offset_lr_delay_mult,
        max_steps=training_args.offset_lr_max_steps)

    self.get_voxel_size()
    self.logger.info(f"Voxel size: {self.voxel_size}")

  def update_learning_rate(self, iteration):
    ''' Learning rate scheduling per step '''
    # lr = self.xyz_scheduler_args(iteration)
    for param_group in self.child_optimizer.param_groups:
      if param_group["name"] == "c_offset":
        param_group['lr'] = self.offset_scheduler_args(iteration)
    for param_group in self.anchor_optimizer.param_groups:
      if param_group["name"] == "a_xyz":
        param_group['lr'] = self.xyz_scheduler_args(iteration)

  def update_params(self, optimizable_tensors):
    self.logger.info(f"Update params, {optimizable_tensors.keys()}")
    # anchor
    if "a_xyz" in optimizable_tensors:
      # self.logger.info("Update anchor xyz, org ({}) -> new ({})".format(
      #     self._anchor_xyz.shape, optimizable_tensors["a_xyz"].shape))
      self._anchor_xyz = optimizable_tensors["a_xyz"]
    if "a_feature" in optimizable_tensors:
      # self.logger.info("Update anchor feature, org ({}) -> new ({})".format(
      #     self._feature.shape, optimizable_tensors["a_feature"].shape))
      self._feature = optimizable_tensors["a_feature"]

    # children
    if "c_offset" in optimizable_tensors:
      # self.logger.info("Update child offset")
      self._offset = optimizable_tensors["c_offset"]
    if "f_dc" in optimizable_tensors:
      # self.logger.info("Update child feature dc")
      self._features_dc = optimizable_tensors["f_dc"]
    if "f_rest" in optimizable_tensors:
      # self.logger.info("Update child feature rest")
      self._features_rest = optimizable_tensors["f_rest"]
    if "c_opacity" in optimizable_tensors:
      # self.logger.info("Update child opacity")
      self._opacity = optimizable_tensors["c_opacity"]
    if "c_scaling" in optimizable_tensors:
      # self.logger.info("Update child scaling")
      self._scaling = optimizable_tensors["c_scaling"]
    if "c_rotation" in optimizable_tensors:
      # self.logger.info("Update child rotation")
      self._rotation = optimizable_tensors["c_rotation"]

  def optimizer_step(self, iteration):
    # if iteration % 50 == 0:
    #   self.logger.info(
    #       "Anchor features grad info: min {}, max {}, mean {}, var {}".format(
    #           self._feature.grad.min(dim=0).values,
    #           self._feature.grad.max(dim=0).values,
    #           self._feature.grad.mean(dim=0), self._feature.grad.var(dim=0)))

    self.child_optimizer.step()
    self.anchor_optimizer.step()
    self.child_optimizer.zero_grad(set_to_none=True)
    self.anchor_optimizer.zero_grad(set_to_none=True)

    self.GraphHelper.step(iteration)

  def mask_gradients(self, anchor_mask):
    """
       Freeze the gradients based on the anchor mask.
       if the anchor_mask is False, the gradients of anchor and its children will be set to zero.
    """
    if self._anchor_xyz.grad is not None:
      self._anchor_xyz.grad[~anchor_mask] = 0.0
    if self._feature.grad is not None:
      self._feature.grad[~anchor_mask] = 0.0

    child_mask = anchor_mask[self._point2anchor]

    child_params = [
        self._offset, self._features_dc, self._features_rest, self._opacity,
        self._scaling, self._rotation
    ]
    for param in child_params:
      if param.grad is not None:
        param.grad[~child_mask] = 0.0

  def save_ply(self, path):
    if not os.path.exists(path.replace("point_cloud.ply", "")):
      os.makedirs(path.replace("point_cloud.ply", ""))
    self.PlyHelper.save_ply(path)

    torch.save((self.capture(), self.ts),
               path.replace("point_cloud.ply", "model.pth"))
    # self.PlyHelper.save_child_points(
    #     path.replace("point_cloud.ply", "child.ply"))

    # self.PlyHelper.save_feature_colors(
    #     path.replace("point_cloud.ply", "feature_vis.ply"))
    self.PlyHelper.save_anchor_points(
        path.replace("point_cloud.ply", "anchor.ply"))

    if self.GraphHelper.is_graph_exist():

      self.logger.info("Saving edges to {}".format(
          path.replace("point_cloud.ply", "graph.pth")))
      self.GraphHelper.save(path.replace("point_cloud.ply", "graph.pth"))

    return

  def load_ply(self, path):
    (data, opt) = torch.load(path.replace("point_cloud.ply", "model.pth"),
                             map_location="cuda")
    self.restore(data, opt)

    if os.path.exists(path.replace("point_cloud.ply", "graph.pth")):
      self.logger.info("Loading graph from {}".format(
          path.replace("point_cloud.ply", "graph.pth")))
      self.GraphHelper.load(path.replace("point_cloud.ply", "graph.pth"))

    return

  def reset_opacity(self, mask=None, opacity_value=0.01):

    opacities_new = inverse_sigmoid(
        torch.min(self.opacity_activation(self._opacity),
                  torch.ones_like(self._opacity) * opacity_value))
    # if true set the opacity to 0.01
    if mask is not None:
      opacities_new[~mask] = self._opacity[~mask]

    optimizable_tensors = self.OptModify.replace_one(self.child_optimizer,
                                                     opacities_new, "c_opacity")
    self._opacity = optimizable_tensors["c_opacity"]

  def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    self.AdControl.grad_threshold = max_grad
    self.AdControl.opacity_threshold = min_opacity
    self.AdControl.extent = extent
    self.AdControl.max_screen_size = max_screen_size
    self.AdControl.control()

  def set_update_filter(self, update_filter):
    self.update_filter = update_filter

  def add_densification_stats(self, viewspace_point_tensor):
    # NOTE: accum tensor shape is (N, 1)
    self.child_offset_gradient_accum[self.update_filter] += torch.norm(
        viewspace_point_tensor.grad[self.update_filter, :2],
        dim=-1,
        keepdim=True)  # (N, 1)
    self.child_offset_denom[self.update_filter] += 1

  def clean_densification_states(self):
    self.child_offset_gradient_accum = torch.zeros_like(
        self.child_offset_gradient_accum)
    self.child_offset_denom = torch.zeros_like(self.child_offset_denom)
    self.max_radii2D = torch.zeros_like(self.max_radii2D)
    self.logger.debug(
        f"check shape: {self._offset.shape} {self.child_offset_gradient_accum.shape} {self.child_offset_denom.shape} {self.max_radii2D.shape}"
    )

  def get_points_from_depth(self, fov_camera, depth, scale=1):
    st = int(max(int(scale / 2) - 1, 0))
    depth_view = depth.squeeze()[st::scale, st::scale]
    rays_d = fov_camera.get_rays(scale=scale)
    depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
    pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
    R = torch.tensor(fov_camera.R).float().cuda()
    T = torch.tensor(fov_camera.T).float().cuda()
    pts = (pts - T) @ R.transpose(-1, -2)
    return pts.reshape(rays_d.shape[0], rays_d.shape[1], 3)

  def search_k_nearest_anchor(self,
                              points,
                              k=3,
                              points_feature=None,
                              feat_weight=0.01):
    # points: (N, 3)
    with torch.no_grad():
      # search knn via pytorch3d
      target = self.scatter_mean_to_anchor(self.get_xyz)
      if points_feature is not None:
        target = torch.cat(
            [target,
             self.segment_activation(self._feature) * feat_weight],
            dim=1)
        points = torch.cat([points, points_feature * feat_weight], dim=1)
      dis, idx, k_anchor = pytorch3d.ops.knn_points(
          points.unsqueeze(0), target.unsqueeze(0), K=k,
          return_nn=True)  # (1, N, k), (1, N, k), (1, N, k, 3)
      return dis.squeeze(0), idx.squeeze(0), k_anchor.squeeze(0)

  def get_voxel_size(self):
    if self.voxel_size <= 1e-6:
      self.logger.info("Voxel size is 0.0, set to default")
      # size_factor = self.update_init_factor // (self.update_hierachy_factor
      #                                           **(self.update_depth - 1))
      size_factor = self.update_init_factor
      voxel_factor = 0.005  # 0.5% of scene size used in our paper
      if self.use_free_scale:
        voxel_factor = 0.01  # use larger factor for saving memory
      self.voxel_size = float(self.spatial_lr_scale) * (
          voxel_factor * self.voxel_size_scale / size_factor)
      self.logger.info(
          f"Voxel size is {self.voxel_size}, size factor is {size_factor}")
    return float(self.voxel_size)

  def remove_invisible_ag(self, views, opacity_threshold, render, pipeline,
                          background):
    """
    We have rethought what is important for scene understanding and editing in engineering projects.
    Of course, the conclusion is that it's just an outer surface.
    This function does not affect actual metrics, but I recommend you use it in engineering projects.
    (Similar geometric constraints also apply)
    """
    torch.cuda.empty_cache()
    time.sleep(1.0)

    self.logger.info(
        f"Remove invisible AG, opacity threshold: {opacity_threshold}")
    # clean optimizer's gradient
    self.child_optimizer.zero_grad(set_to_none=True)
    self.anchor_optimizer.zero_grad(set_to_none=True)

    opacity_grad = torch.zeros_like(self._opacity).requires_grad_(False)
    self.logger.error(
        f"Opacity grad min: {opacity_grad.min()}, max: {opacity_grad.max()}, mean: {opacity_grad.mean()}, var: {opacity_grad.var()}"
    )
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
      render_pkg = render(view, self, pipeline, background)

      opacity_map = render_pkg["alpha"]

      loss = opacity_map.sum()
      loss.backward()

      opacity_grad += torch.abs(self._opacity.grad)

      opacity_grad = torch.clamp(opacity_grad, min=0.0, max=1.0)

      # clean optimizer's gradient
      self.child_optimizer.zero_grad(set_to_none=True)
      self.anchor_optimizer.zero_grad(set_to_none=True)

    # get the opacity gradient
    # opacity_grad = self._opacity.grad
    self.logger.error(
        f"Opacity grad min: {opacity_grad.min()}, max: {opacity_grad.max()}, mean: {opacity_grad.mean()}, var: {opacity_grad.var()}"
    )

    if True:
      # plot the opacity gradient
      import matplotlib.pyplot as plt
      import matplotlib.cm as cm

      plt.figure(figsize=(10, 5))
      plt.hist(opacity_grad.cpu().numpy(), bins=100, color='blue')
      plt.savefig("opacity_grad.png")

    selected_gs = opacity_grad < opacity_threshold

    self.AdControl.gaussian_prune(0, 0, selected_gs.view(-1))
    self.AdControl.anchor_prune()

    torch.cuda.empty_cache()
    time.sleep(1.0)
