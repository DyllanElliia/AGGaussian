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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import matplotlib.pyplot as plt
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.log_utils import *

from utils.math_utils import feature_map_to_rgb

# def depth_to_image(depth):
#   return (depth - depth.min()) / (depth.max() - depth.min())


def depth_to_image(depth_map):
  """
    Convert a single-channel depth map to a 3-channel RGB image using the Viridis colormap.
    
    Args:
        depth_map (torch.Tensor): Tensor of shape (1, H, W) representing the depth map.

    Returns:
        torch.Tensor: Tensor of shape (3, H, W) representing the RGB depth map.
    """
  assert depth_map.dim() == 3 and depth_map.shape[
      0] == 1, "Input depth map must have shape (1, H, W)"
  device = depth_map.device
  # Normalize depth values to range [0, 1]
  depth_min, depth_max = depth_map.min(), depth_map.max()
  depth_normalized = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)

  # Convert to numpy and apply colormap
  depth_numpy = depth_normalized.squeeze(0).cpu().numpy()
  colormap = plt.get_cmap('viridis')
  colored_depth = colormap(depth_numpy)[:, :, :3]

  # Convert back to torch tensor and rearrange to (3, H, W)
  rgb_tensor = torch.from_numpy(colored_depth).to(device).permute(2, 0, 1)

  return rgb_tensor


def render_set(model_path, name, iteration, views, gaussians, pipeline,
               background):

  render_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                             "renders")
  semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                               "semantic")
  depth_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                            "depth")
  depth_dist_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                                 "depth_dist")

  gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

  makedirs(render_path, exist_ok=True)
  makedirs(semantic_path, exist_ok=True)
  makedirs(depth_path, exist_ok=True)
  makedirs(gts_path, exist_ok=True)
  makedirs(depth_dist_path, exist_ok=True)

  for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
    render_pkg = render(view, gaussians, pipeline, background)
    gt = view.original_image[0:3, :, :]
    torchvision.utils.save_image(
        gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    torchvision.utils.save_image(
        render_pkg["render"],
        os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
    # torchvision.utils.save_image(
    #     feature_map_to_rgb(render_pkg["feature"]),
    #     os.path.join(semantic_path, '{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(
        (render_pkg["feature"] * 0.5 + 0.5).clamp(0, 1),
        os.path.join(semantic_path, '{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(
        depth_to_image(render_pkg["depth"]),
        os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    torchvision.utils.save_image(
        depth_to_image(render_pkg["depth_distortion"]),
        os.path.join(depth_dist_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool):
  logger = get_loguru_logger(os.path.join(args.model_path))
  with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree,
                              init_child_num=5,
                              use_free_scale=dataset.use_free_scale,
                              logger=logger)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
      render_set(dataset.model_path, "train", scene.loaded_iter,
                 scene.getTrainCameras(), gaussians, pipeline, background)

    if not skip_test:
      render_set(dataset.model_path, "test", scene.loaded_iter,
                 scene.getTestCameras(), gaussians, pipeline, background)


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Testing script parameters")
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument("--iteration", default=-1, type=int)
  parser.add_argument("--skip_train", action="store_true")
  parser.add_argument("--skip_test", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  args = get_combined_args(parser)
  print("Rendering " + args.model_path)

  # Initialize system state (RNG)
  safe_state(args.quiet)

  render_sets(model.extract(args), args.iteration, pipeline.extract(args),
              args.skip_train, args.skip_test)
