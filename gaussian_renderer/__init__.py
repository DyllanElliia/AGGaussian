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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
from scene.anchor_gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    override_semantics=None,
    override_xyz=None,
    override_cov3D=None,
    override_opacity=None,
    render_ellipsoid=False,
    opacity_modifier=None,
    opacity_mask=None,
    render_pca=False,
    lang_mode=False,
):
  """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    
    scaling_modifier: float, optional, use a smaller value to render the Gaussians smaller.
    override_color: tensor, optional, if provided, will override the colors of the Gaussians.
    render_ellipsoid: bool, optional, if True, will render the ellipsoids of the Gaussians.
    opacity_modifier: float, optional, if provided, will override the opacity of the Gaussians.
    
    """

  # Set up rasterization configuration
  tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
  tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
  if render_ellipsoid:
    bg_color = 0.0 * torch.ones_like(bg_color).to("cuda")

  raster_settings = GaussianRasterizationSettings(
      image_height=int(viewpoint_camera.image_height),
      image_width=int(viewpoint_camera.image_width),
      tanfovx=tanfovx,
      tanfovy=tanfovy,
      bg=bg_color,
      scale_modifier=scaling_modifier,
      viewmatrix=viewpoint_camera.world_view_transform,
      projmatrix=viewpoint_camera.full_proj_transform,
      sh_degree=pc.active_sh_degree,
      campos=viewpoint_camera.camera_center,
      prefiltered=False,
      debug=pipe.debug,
      render_ellipsoid=render_ellipsoid)

  rasterizer = GaussianRasterizer(raster_settings=raster_settings)

  means3D = None
  if override_xyz is None:
    means3D = pc.get_xyz
  else:
    means3D = override_xyz

  # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
  screenspace_points = torch.zeros_like(
      means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
  try:
    screenspace_points.retain_grad()
  except:
    pass

  means2D = screenspace_points
  opacity = None
  if override_opacity is None:
    opacity = pc.get_opacity
  else:
    opacity = override_opacity

  if lang_mode:
    means3D = means3D.detach()
    # means2D = means2D.detach()
    opacity = opacity.detach()

  if opacity_mask is not None:
    opacity = opacity * opacity_mask

  if opacity_modifier is not None:
    opacity = opacity * 0.0 + opacity_modifier

  # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
  # scaling / rotation by the rasterizer.
  scales = None
  rotations = None
  cov3D_precomp = None
  if override_cov3D is None:
    if pipe.compute_cov3D_python:
      cov3D_precomp = pc.get_covariance(scaling_modifier)
      if lang_mode:
        cov3D_precomp = cov3D_precomp.detach()
    else:
      scales = pc.get_scaling
      rotations = pc.get_rotation
      if lang_mode:
        scales = scales.detach()
        rotations = rotations.detach()
  else:
    cov3D_precomp = override_cov3D

  # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
  # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
  shs = None
  colors_precomp = None
  if override_color is None:
    if pipe.convert_SHs_python:
      shs_view = pc.get_features.transpose(1, 2).view(-1, 3,
                                                      (pc.max_sh_degree + 1)**2)
      dir_pp = (
          means3D -
          viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
      dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
      sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
      colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
      if lang_mode:
        colors_precomp = colors_precomp.detach()
    else:
      shs = pc.get_features
      if lang_mode:
        shs = shs.detach()
  else:
    colors_precomp = override_color
    # colors_precomp.register_hook(
    #     lambda grad: print(f"Gradient of colors_precomp: {grad.max()}"))

  semantics = None
  if override_semantics is None:
    # print("Using semantics from GaussianModel")
    override_semantics = pc.get_semantics
    if lang_mode:
      override_semantics = override_semantics.detach()
    semantics = override_semantics
  else:
    semantics = override_semantics
    # semantics.register_hook(
    #     lambda grad: print(f"Gradient of semantics: {grad.max()}"))

  # Rasterize visible Gaussians to image, obtain their radii (on screen).
  rendered_image, rendered_sem, radii, depth, alpha, distortion = rasterizer(
      means3D=means3D,
      means2D=means2D,
      shs=shs,
      colors_precomp=colors_precomp,
      semantics=semantics,
      opacities=opacity,
      scales=scales,
      rotations=rotations,
      cov3D_precomp=cov3D_precomp)
  # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
  # They will be excluded from value updates used in the splitting criteria.
  rendered_sem /= alpha
  rendered_sem[:, alpha.squeeze(0) < 0.1] = 0.0
  return {
      "render": rendered_image,
      "feature": rendered_sem,
      "depth": depth,
      "alpha": alpha,
      "depth_distortion": distortion[0:1, :, :],
      "feature_distortion": distortion[1:4, :, :],
      "viewspace_points": screenspace_points,
      "visibility_filter": radii > 0,
      "radii": radii
  }
