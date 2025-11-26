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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal


class MaskBitpack:
  """(M,H,W) bool  <->  (1,H,W) int32/int64 packing/unpacking"""
  """
  A simple class to bit-pack multiple binary masks into a single integer mask.
  This is useful for saving memory when you input a lot of images.
  We didn't use it in our paper, but feel free to use it in your project!
  """

  def __init__(self, bits: int = 32):
    assert bits in (8, 16, 32, 64)
    self.bits = bits
    switcher = {
        8: torch.uint8,
        16: torch.int16,
        32: torch.int32,
        64: torch.int64
    }
    self.dtype = switcher.get(bits, torch.int32)
    self.M_num = None
    self.packed = None
    self.shape = None

  @torch.no_grad()
  def encode(self, mask: torch.Tensor) -> torch.Tensor:
    """mask: (M,H,W) bool -> (1,H,W) int32/int64, and record M_num"""
    M, H, W = mask.shape
    self.shape = mask.shape
    # assert M <= self.bits
    self.M_num = M
    device = mask.device

    cls = torch.arange(1, M + 1, dtype=self.dtype, device=device).view(M, 1, 1)
    assert (mask.sum(0) <= 1).all(), "mask overlaps -> aliasing"
    self.packed = (mask.to(self.dtype) * cls).amax(dim=0)  # (H,W) ∈ [0..M]

  @torch.no_grad()
  def decode(self) -> torch.Tensor:
    """packed: (1,H,W) int32/int64 -> (M,H,W) bool (using self.M_num)"""
    M = self.M_num
    H, W = self.packed.shape
    device = self.packed.device

    cls = torch.arange(1, M + 1, device=device, dtype=self.dtype).view(M, 1, 1)
    out = (self.packed.view(1, H, W) == cls).to(torch.bool)  # (M,H,W)

    return out

  def to(self, device):
    mb = MaskBitpack(bits=self.bits)
    mb.M_num = self.M_num
    mb.dtype = self.dtype
    mb.shape = self.shape
    if self.packed is not None:
      mb.packed = self.packed.to(device)

    return mb

  def cuda(self):
    return self.to('cuda')

  def cpu(self):
    return self.to('cpu')

  def to_dense(self):
    return self.decode()


class Camera(nn.Module):

  def __init__(
      self,
      colmap_id,
      R,
      T,
      FoVx,
      FoVy,
      image,
      image_width_height,
      #  mask_sparse,
      gt_alpha_mask,
      image_name,
      mask_sparse_path,
      clip_feat_path,
      uid,
      trans=np.array([0.0, 0.0, 0.0]),
      scale=1.0,
      data_device="cuda"):
    super(Camera, self).__init__()

    self.uid = uid
    self.colmap_id = colmap_id
    self.R = R
    self.T = T
    self.FoVx = FoVx
    self.FoVy = FoVy
    self.resolution = image_width_height  # (image_width, image_height)
    self.Fx = fov2focal(self.FoVx, self.resolution[0])
    self.Fy = fov2focal(self.FoVy, self.resolution[1])
    self.Cx = 0.5 * self.resolution[0]
    self.Cy = 0.5 * self.resolution[1]

    self.image_name = image_name
    self.mask_sparse_name = image_name

    try:
      self.data_device = torch.device(data_device)
    except Exception as e:
      print(e)
      print(
          f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
      )
      self.data_device = torch.device("cuda")

    self.original_image = image.clamp(0.0, 1.0).to(self.data_device)

    # self.original_mask = mask
    self.mask_sparse_path = mask_sparse_path
    self.clip_feat_path = clip_feat_path
    try:
      self.original_mask_sparse = torch.load(self.mask_sparse_path,
                                             weights_only=True)
      self.clip_feat = torch.load(clip_feat_path, weights_only=True)
    except:
      self.original_mask_sparse = None
      self.clip_feat = None

    self.image_width = self.original_image.shape[2]
    self.image_height = self.original_image.shape[1]

    if gt_alpha_mask is not None:
      self.original_image *= gt_alpha_mask.to(self.data_device)
    else:
      self.original_image *= torch.ones(
          (1, self.image_height, self.image_width), device=self.data_device)

    self.original_image = self.original_image.cpu()

    self.zfar = 100.0
    self.znear = 0.01

    self.trans = trans
    self.scale = scale

    self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans,
                                                            scale)).transpose(
                                                                0, 1).cuda()
    self.projection_matrix = getProjectionMatrix(znear=self.znear,
                                                 zfar=self.zfar,
                                                 fovX=self.FoVx,
                                                 fovY=self.FoVy).transpose(
                                                     0, 1).cuda()
    self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(
        self.projection_matrix.unsqueeze(0))).squeeze(0)
    self.camera_center = self.world_view_transform.inverse()[3, :3]

  def get_rays(self, scale=1.0):
    W, H = int(self.resolution[0] / scale), int(self.resolution[1] / scale)
    ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    rays_d = torch.stack([(ix - self.Cx / scale) / self.Fx * scale,
                          (iy - self.Cy / scale) / self.Fy * scale,
                          torch.ones_like(ix)], -1).float().cuda()
    return rays_d


class MiniCam:

  def __init__(self, width, height, fovy, fovx, znear, zfar,
               world_view_transform, full_proj_transform):
    self.image_width = width
    self.image_height = height
    self.FoVy = fovy
    self.FoVx = fovx
    self.znear = znear
    self.zfar = zfar
    self.world_view_transform = world_view_transform
    self.full_proj_transform = full_proj_transform
    view_inv = torch.inverse(self.world_view_transform)
    self.camera_center = view_inv[3][:3]
