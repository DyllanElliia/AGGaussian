import os

import torch
import torch.nn as nn

import numpy as np
from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2
from einops import rearrange, repeat

from .. import GaussianModel
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p


class PlyHelper:

  def __init__(self, logger, model: GaussianModel):
    self.logger = logger
    self.model = model

    self.voxel_size = getattr(self.model, "voxel_size", 0.01)

  def save_points_colors(self, path, points, colors, mask=None):
    if mask is not None:
      points = points[mask]
      colors = colors[mask]
    points = torch.cat((points, colors),
                       dim=1).transpose(0, 1).detach().cpu().numpy()
    vertex = np.core.records.fromarrays(points,
                                        names='x, y, z, red, green, blue',
                                        formats='f4, f4, f4, u1, u1, u1')
    ply_data = PlyData([PlyElement.describe(vertex, 'vertex')], text=True)
    ply_data.write(path)

  def save_points_edges_colors(self,
                               path,
                               points,
                               edges,
                               point_colors,
                               edge_weight=None):
    points = torch.cat((points, point_colors),
                       dim=1).transpose(0, 1).detach().cpu().numpy()
    vertex = np.core.records.fromarrays(points,
                                        names='x, y, z, red, green, blue',
                                        formats='f4, f4, f4, u1, u1, u1')
    if edge_weight is not None:
      edge_weight = edge_weight * 255
      edges = torch.cat((edges, edge_weight, edge_weight, edge_weight),
                        dim=1).transpose(0, 1).detach().cpu().numpy()
      edge = np.core.records.fromarrays(
          edges,
          names='vertex1, vertex2, red, green, blue',
          formats='i4, i4, u1, u1, u1')
    else:
      edges = edges.transpose(0, 1).detach().cpu().numpy()
      edge = np.core.records.fromarrays(edges,
                                        names='vertex1, vertex2',
                                        formats='i4, i4')

    ply_data = PlyData([
        PlyElement.describe(vertex, 'vertex'),
        PlyElement.describe(edge, 'edge')
    ],
                       text=True)
    ply_data.write(path)

  def save_child_points(self, path, opacity_threshold=0.1):
    with torch.no_grad():
      op = (self.model.get_opacity).view(-1)
      points = self.model.get_xyz
      colors = (self.model.get_semantics * .5 + .5) * 255.
      mask = op > opacity_threshold
      self.save_points_colors(path, points, colors, mask)

  def save_anchor_points(self, path, f=None):
    with torch.no_grad():
      points = self.model._anchor_xyz
      colors = ((self.model.segment_activation(self.model._feature)
                 if f == None else f) * .5 + .5) * 255.
      self.save_points_colors(path, points, colors)

  def save_feature_colors(self, path, f=None):
    with torch.no_grad():
      points = self.model.segment_activation(
          self.model._feature) if f == None else f
      colors = (points * .5 + .5) * 255.
      self.save_points_colors(path, points, colors)

  def save_anchor_points_edges(self, path, f=None):
    with torch.no_grad():
      points = self.model._anchor_xyz
      point_colors = ((self.model.segment_activation(self.model._feature)
                       if f == None else f) * .5 + .5) * 255.
      edges = self.model.GraphHelper.edges

      edge_weight = self.model.GraphHelper.compute_all_edge_weight(
          lambda x: self.model.GraphHelper.cal_edge_weight(x))
      self.save_points_edges_colors(path, points, edges, point_colors,
                                    edge_weight)

  def voxelize_sample(self, data=None, voxel_size=0.01):
    # np.random.shuffle(data)
    data, idx = np.unique(np.round(data / voxel_size),
                          axis=0,
                          return_index=True)

    return data * voxel_size, idx

  def create_from_pcd(self, pcd, spatial_lr_scale):
    self.logger.info("Creating Gaussian model from point cloud")
    # spatial_lr_scale <- scene_info.nerf_normalization["radius"]
    # spatial_lr_scale
    self.model.spatial_lr_scale = spatial_lr_scale
    self.logger.info("Spatial LR scale: {}".format(spatial_lr_scale))
    self.model.voxel_size = self.model.get_voxel_size()
    # self.logger.info("Voxel size: {}".format(self.model.voxel_size))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    points, idx = self.voxelize_sample(points, self.model.voxel_size)
    colors = colors[idx]
    # ours is neural-free, so we don't need to shuffle them
    # # shuffle points and colors idx
    # idx = np.random.permutation(points.shape[0])
    # points = points[idx]
    # colors = colors[idx]

    fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
    fused_color = RGB2SH(torch.tensor(np.asarray(colors)).float().cuda())
    del points, colors, idx

    features = torch.zeros((fused_color.shape[0], 3,
                            (self.model.max_sh_degree + 1)**2)).float().cuda()
    features[:, :3, 0] = fused_color
    features[:, 3:, 1:] = 0.0

    self.logger.info("Number of points at initialisation : {}".format(
        fused_point_cloud.shape[0]))

    dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001)
    # dist2 = torch.clamp_max(dist2, 0.02 * spatial_lr_scale)
    # dist2 = torch.clamp_max(dist2,
    #                         self.model.voxel_size * self.model.update_init_factor * 4)
    dist = torch.sqrt(dist2)
    # scales = torch.log(dist)[..., None].repeat(1, 3)
    rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    rots[:, 0] = 1

    opacities = self.model.inverse_opacity_activation(0.01 * torch.ones(
        (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    # TODO: How to initialise anchor radii?
    # anchor_r = dist.clone() * 1.5
    anchor_r = dist.clone()
    # anchor_r = torch.clamp(anchor_r, 0.001 * spatial_lr_scale,
    #                        0.05 * spatial_lr_scale)
    if self.model.use_free_scale:
      # anchor_r = torch.clamp_max(anchor_r, 0.05 * spatial_lr_scale)
      anchor_r = torch.clamp_max(
          anchor_r, self.model.update_init_factor * self.model.voxel_size)
      anchor_r = torch.floor(
          anchor_r / self.model.voxel_size) * self.model.voxel_size

    self.logger.info("Mean radius of anchor points : {}".format(
        anchor_r.mean()))
    self.logger.info("Min and Max radius of anchor points : {} {}".format(
        anchor_r.min(), anchor_r.max()))

    # TODO: Check if this is correct
    self.model._anchor_xyz = nn.Parameter(
        fused_point_cloud.contiguous().requires_grad_(True))
    self.model._anchor_radii = rearrange(anchor_r, "n -> n 1").contiguous()
    # Init point2anchor
    self.model._point2anchor = repeat(torch.arange(
        self.model._anchor_xyz.shape[0], dtype=torch.int32, device="cuda"),
                                      "n -> (n c)",
                                      c=self.model.init_child_num).contiguous()

    # repeat features, scales, rots, opacities for each child
    features = repeat(features,
                      'n sh d -> (n c) sh d',
                      c=self.model.init_child_num).contiguous()
    scales = torch.clamp_min(dist[..., None],
                             0.001 * spatial_lr_scale).repeat(1, 3)
    scales = repeat(scales, 'n d -> (n c) d',
                    c=self.model.init_child_num).contiguous()
    scales = self.model.scaling_inverse_activation(0.5 * scales)
    rots = repeat(rots, 'n q -> (n c) q',
                  c=self.model.init_child_num).contiguous()
    opacities = repeat(opacities, 'n o -> (n c) o',
                       c=self.model.init_child_num).contiguous()

    # init segment feature (N, 3)
    # self.model._feature = nn.Parameter(
    #     torch.zeros((fused_point_cloud.shape[0], 3),
    #                 device=fused_point_cloud.device))
    self.model._feature = nn.Parameter(
        torch.ones((fused_point_cloud.shape[0], 3),
                   device=fused_point_cloud.device))

    # self.model._offset = nn.Parameter(
    #     torch.randn((fused_point_cloud.shape[0] * self.model.init_child_num, 3),
    #                 device=fused_point_cloud.device) / 3.0)
    # scales = scales / (self.model.init_child_num)
    self.model._offset = nn.Parameter(
        torch.zeros((fused_point_cloud.shape[0] * self.model.init_child_num, 3),
                    device=fused_point_cloud.device))

    self.model._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
        1, 2).contiguous().requires_grad_(True))
    self.model._features_rest = nn.Parameter(features[:, :, 1:].transpose(
        1, 2).contiguous().requires_grad_(True))
    self.model._scaling = nn.Parameter(scales.requires_grad_(True))
    self.model._rotation = nn.Parameter(rots.requires_grad_(True))
    self.model._opacity = nn.Parameter(opacities.requires_grad_(True))

    self.logger.info("Number of points after initialisation : {}".format(
        self.model._offset.shape[0]))
    self.logger.info("Anchor shape : {}".format(self.model._anchor_xyz.shape))
    self.logger.info("Index shape : {}".format(self.model._point2anchor.shape))
    self.logger.info("SH feature shape : {}".format(
        self.model.get_features.shape))

    self.model.max_radii2D = torch.zeros((self.model.get_xyz.shape[0]),
                                         device="cuda")
    torch.cuda.empty_cache()

  def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self.model._features_dc.shape[1] *
                   self.model._features_dc.shape[2]):
      l.append('f_dc_{}'.format(i))
    for i in range(self.model._features_rest.shape[1] *
                   self.model._features_rest.shape[2]):
      l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self.model._scaling.shape[1]):
      l.append('scale_{}'.format(i))
    for i in range(self.model._rotation.shape[1]):
      l.append('rot_{}'.format(i))
    return l

  def save_ply(self, path):
    mkdir_p(os.path.dirname(path))

    xyz = self.model.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = self.model._features_dc.detach().transpose(
        1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = self.model._features_rest.detach().transpose(
        1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = self.model._opacity.detach().cpu().numpy()
    scale = torch.log(self.model.get_scaling.detach()).cpu().numpy()
    rotation = self.model._rotation.detach().cpu().numpy()

    dtype_full = [
        (attribute, 'f4') for attribute in self.construct_list_of_attributes()
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
