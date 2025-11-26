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

import os
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from random import randint
from utils.loss_utils import l1_loss, ssim, mask_constrastive_loss, mask_constrastive_loss_2, og_mask_constrastive_loss, og_mask_constrastive_loss_org, distortion_mask_loss, lang_distil_loss
from utils.math_utils import feature_map_to_rgb
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from einops import rearrange, repeat

try:
  from torch.utils.tensorboard import SummaryWriter
  TENSORBOARD_FOUND = True
except ImportError:
  TENSORBOARD_FOUND = False

if TENSORBOARD_FOUND:
  print("Tensorboard found")
else:
  print("Tensorboard not found")


def training(dataset, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, load_iteration, debug_from):
  first_iter = 0
  tb_writer, logger = prepare_output_and_logger(dataset, use_logger=True)
  logger.remove()
  gaussians = GaussianModel(
      dataset.sh_degree,
      init_child_num=dataset.init_child_num,
      voxel_size=dataset.voxel_size,
      voxel_size_scale=dataset.voxel_size_scale,
      update_depth=dataset.update_depth,
      update_init_factor=dataset.update_init_factor,
      update_hierachy_factor=dataset.update_hierachy_factor,
      use_free_scale=dataset.use_free_scale,
      logger=logger,
      training_args=opt)
  scene = Scene(dataset, gaussians, load_iteration=load_iteration)
  gaussians.training_setup(opt)
  gaussians.AdControl.stds_scale = opt.stds_scale
  if load_iteration is not None:
    first_iter = load_iteration
  if checkpoint:
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)

  logger.info("Background color: {}".format(
      "white" if dataset.white_background else "black"))
  bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
  background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  iter_start = torch.cuda.Event(enable_timing=True)
  iter_end = torch.cuda.Event(enable_timing=True)

  viewpoint_stack = None
  ema_loss_for_log = 0.0
  progress_bar = tqdm(range(first_iter, opt.iterations),
                      desc="Training progress")
  first_iter += 1
  logger.info(f"Start training from iteration {first_iter}")
  for iteration in range(first_iter, opt.iterations + 1):
    iter_start.record()

    gaussians.update_learning_rate(iteration)

    # Every 1000 its we increase the levels of SH up to a maximum degree
    if iteration % 1000 == 0:
      gaussians.oneupSHdegree()

    # Pick a random Camera
    if not viewpoint_stack:
      viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

    # Render
    if (iteration - 1) == debug_from:
      pipe.debug = True

    bg = torch.rand((3), device="cuda") if opt.random_background else background

    render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
    image, viewspace_point_tensor, visibility_filter, radii, semantic = render_pkg[
        "render"], render_pkg["viewspace_points"], render_pkg[
            "visibility_filter"], render_pkg["radii"], render_pkg["feature"]

    # semantic = classifier(semantic.unsqueeze(0)).squeeze(0)

    # Image loss
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
        1.0 - ssim(image, gt_image))

    # loss += 0.1 * l1_loss(semantic, gt_image)

    # Gaussian loss
    gaussians.set_update_filter(visibility_filter)
    lossTotal, lossList = gaussians.get_loss()

    loss += lossTotal

    lambda_depth_dist = opt.lambda_depth_dist if iteration > 10_000 else opt.lambda_depth_dist * 0.0
    lambda_feat_dist = opt.lambda_feat_dist if iteration > 20_000 else opt.lambda_feat_dist * 1e-3

    if True and iteration > 5000 and viewpoint_cam.clip_feat is not None:
      # Mask loss

      mask = viewpoint_cam.original_mask_sparse.cuda().to_dense()

      mask_loss = og_mask_constrastive_loss_org(semantic,
                                                mask,
                                                alpha=opt.lambda_m_smooth,
                                                beta=opt.lambda_m_contrast,
                                                gamma=0.0,
                                                epsilon=1.0,
                                                mode="l2",
                                                feat_norm_switch=False)

      loss += 0.1 * mask_loss
      mask_loss = mask_loss.item()

      loss_d_d = lambda_depth_dist * render_pkg["depth_distortion"].mean()
      loss += loss_d_d
    else:
      mask_loss = 0.0

      loss_d_d = lambda_depth_dist * render_pkg["depth_distortion"].mean()

      loss_f_d = 0.0

      loss += loss_d_d + loss_f_d

    # opacity loss can encourage the gaussian to extend to the empty space, which sfm does not sample.
    # it can improve the reconstruction quality. but it is not necessary for segmentation task.
    if True and iteration > 3_000:
      loss += 0.1 * (1 - render_pkg["alpha"]).mean()

    loss.backward()

    iter_end.record()

    # classifier_optim.step()
    # classifier_optim.zero_grad()

    with torch.no_grad():
      # Progress bar
      ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
      if iteration % 10 == 0:
        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
        progress_bar.update(10)
      if iteration == opt.iterations:
        progress_bar.close()

      # Log and save
      # training_report(tb_writer,
      #                 iteration,
      #                 Ll1,
      #                 loss,
      #                 l1_loss,
      #                 mask_loss,
      #                 iter_start.elapsed_time(iter_end),
      #                 testing_iterations,
      #                 scene,
      #                 render, (pipe, background),
      #                 gaussians.get_xyz.shape[0],
      #                 lossList=lossList)
      if (iteration in saving_iterations):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        last_save_iter = iteration
        scene.save(iteration)

      # Densification
      if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
          size_threshold = 40 if iteration > opt.opacity_reset_interval else None
          opacity_threshold = 0.0099  # 0.005
          gaussians.densify_and_prune(opt.densify_grad_threshold,
                                      opacity_threshold, scene.cameras_extent,
                                      size_threshold)

        if iteration % opt.opacity_reset_interval == 0 or (
            dataset.white_background and iteration == opt.densify_from_iter):
          gaussians.reset_opacity()

      # Optimizer step
      if iteration < opt.iterations:
        gaussians.optimizer_step(iteration)
        torch.cuda.empty_cache()

      # Debug info
      if False and iteration % 100 == 0:
        path = os.path.join(scene.model_path, "debug")
        if not os.path.exists(path):
          os.makedirs(path)
        # save image and semantic
        cat_i_s = torch.cat([image, semantic / 2 + .5], dim=2)
        i_d = render_pkg["depth"]
        # i_d = (i_d - i_d.min()) / (i_d.max() - i_d.min())
        i_d = i_d / i_d.max()
        cat_d_a = torch.cat([render_pkg["alpha"], i_d], dim=2)
        cat_img = torch.cat(
            [cat_i_s, repeat(cat_d_a, "1 h w -> d h w", d=3)], dim=1)

        i_d_d = render_pkg["depth_distortion"]
        i_d_d = (i_d_d - i_d_d.min()) / (i_d_d.max() - i_d_d.min() + 1e-8)
        i_d_d = repeat(torch.clamp_max(i_d_d, 1.0), "1 h w -> d h w", d=3)
        cat_d_f = torch.cat([render_pkg["feature_distortion"].abs(), i_d_d],
                            dim=1)
        cat_img = torch.cat([cat_img, cat_d_f], dim=2)

        downsampled_img = torch.nn.functional.interpolate(
            cat_img.unsqueeze(0),
            scale_factor=0.5,
            mode='bilinear',
            align_corners=False).squeeze(0)

        torchvision.utils.save_image(downsampled_img,
                                     os.path.join(path, f"img_{iteration}.png"))

        if iteration > opt.graph_propagation_begin and iteration % 1000 == 0:
          plt.figure(figsize=(10, 8))
          # subfig 1
          plt.subplot(2, 1, 1)
          plt.hist(gaussians.GraphHelper.compute_all_edge_weight(
              lambda e: gaussians.GraphHelper.cal_edge_weight(e)).cpu().numpy(),
                   bins=100)
          plt.xlabel("Edge Weight")
          plt.ylabel("Frequency")

          # subfig 2
          plt.subplot(2, 1, 2)
          plt.hist(gaussians.GraphHelper.compute_all_edge_feature_dist().cpu().
                   numpy(),
                   bins=100)
          plt.xlabel("Edge Feature Distance")
          plt.ylabel("Frequency")
          plt.tight_layout()
          plt.savefig(os.path.join(path, f"edge_weight_hist.png"))

      if (iteration in checkpoint_iterations):
        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save((gaussians.capture(), iteration),
                   scene.model_path + "/chkpnt" + str(iteration) + ".pth")
    if iteration > opt.densify_from_iter and iteration in [
        14_990,
        opt.graph_propagation_begin + 1,
    ]:
      gaussians.remove_invisible_ag(scene.getTrainCameras(), 1e-3, render, pipe,
                                    bg)
    if iteration == opt.graph_propagation_begin + 1:

      gaussians.GraphHelper.init_model()
      with torch.no_grad():
        gaussians.GraphHelper.build_graph(opt.graph_prop_scale,
                                          link_isolated=False)

      gaussians.GraphHelper.train()
  logger.info("Training complete (model saved to {})".format(args.model_path))
  # del loss
  torch.cuda.empty_cache()
  # merge anchor
  gaussians.GraphHelper.build_graph(edge_dist_scale=opt.graph_merge_scale,
                                    link_isolated=False)
  s_a_xyz, s_a_f, s_a_edge, a2sa_idx = gaussians.GraphHelper.get_super_anchor(
      opt)
  logger.info(f"super anchor num: {s_a_xyz.shape[0]}")
  logger.info(f"anchor num: {gaussians.get_xyz.shape[0]}")
  logger.info(f"anchor2superanchor idx: {a2sa_idx.shape[0]}")
  logger.info(f"super anchor graph edge num: {s_a_edge.shape[0]}")
  logger.info(
      f"anchor feature max min mean: {s_a_f.max()} {s_a_f.min()} {s_a_f.mean()}"
  )
  logger.info(
      f"device of s a {s_a_xyz.device} {s_a_f.device} {s_a_edge.device} {a2sa_idx.device}"
  )
  gaussians.PlyHelper.save_points_edges_colors(
      os.path.join(scene.model_path, "test_super_anchor.ply"), s_a_xyz,
      s_a_edge, (s_a_f * 0.5 + 0.5) * 255)
  gaussians.LangHelper.init_sa(s_a_xyz, s_a_f, a2sa_idx)
  # return

  del s_a_xyz, s_a_f, a2sa_idx
  torch.cuda.empty_cache()
  gaussians.LangHelper.init_neural_mapping(
      scene.getTrainCameras()[0].clip_feat.shape[1])

  gaussians.LangHelper.train()
  del viewpoint_stack

  if gaussians.LangHelper.need_train() == False:
    gaussians.LangHelper.model_path = scene.model_path
    gaussians.LangHelper.match(render, scene, pipe, background)

    gaussians.LangHelper.save(scene.model_path)
    scene.save(opt.iterations)

  logger.info("Training complete (model saved to {})".format(scene.model_path))


def prepare_output_and_logger(args, use_logger=False):
  if not args.model_path:
    if os.getenv('OAR_JOB_ID'):
      unique_str = os.getenv('OAR_JOB_ID')
    else:
      unique_str = str(uuid.uuid4())
    args.model_path = os.path.join("./output/", unique_str[0:10])

  # Set up output folder
  print("Output folder: {}".format(args.model_path))
  os.makedirs(args.model_path, exist_ok=True)
  with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
    cfg_log_f.write(str(Namespace(**vars(args))))

  # Create Tensorboard writer
  tb_writer = None
  if TENSORBOARD_FOUND:
    tb_writer = SummaryWriter(args.model_path)
  else:
    print("Tensorboard not available: not logging progress")
  if use_logger:
    from utils.log_utils import get_logger, get_loguru_logger
    # logger = Logger(os.path.join(args.model_path, "train.log"))
    # logger = get_logger(log_path=os.path.join(args.model_path, "train.log"))
    logger = get_loguru_logger(
        log_path=os.path.join(args.model_path, "train.log"))
    # logger.info("Training started")
    # logger.info("Arguments: {}".format(args))
    return tb_writer, logger
  else:
    return tb_writer


def training_report(tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    mask_loss,
                    elapsed,
                    testing_iterations,
                    scene: Scene,
                    renderFunc,
                    renderArgs,
                    gauss_num=None,
                    lossList=None):
  if tb_writer:
    tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    tb_writer.add_scalar('train_loss_patches/mask_loss', mask_loss, iteration)
    tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(),
                         iteration)
    if lossList:
      for i, name in enumerate(lossList):
        tb_writer.add_scalar('train_loss_patches/loss_{}'.format(name),
                             lossList[name], iteration)
    tb_writer.add_scalar('iter_time', elapsed, iteration)

  if gauss_num:
    tb_writer.add_scalar('scene/gaussian_num', gauss_num, iteration)
    tb_writer.add_scalar('scene/anchor_num',
                         scene.gaussians._anchor_xyz.shape[0], iteration)

  # Report test and samples of training set
  if iteration in testing_iterations:
    torch.cuda.empty_cache()
    validation_configs = ({
        'name': 'test',
        'cameras': scene.getTestCameras()
    }, {
        'name':
            'train',
        'cameras': [
            scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
            for idx in range(5, 30, 5)
        ]
    })

    for config in validation_configs:
      if config['cameras'] and len(config['cameras']) > 0:
        l1_test = 0.0
        psnr_test = 0.0
        for idx, viewpoint in enumerate(config['cameras']):
          image_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
          image = torch.clamp(image_pkg["render"], 0.0, 1.0)
          # normalize depth map to [0, 1]
          image_depth = image_pkg["depth"]
          image_depth = (image_depth - image_depth.min()) / (image_depth.max() -
                                                             image_depth.min())
          image_sem = torch.clamp(image_pkg["feature"] / 2 + .5, 0.0, 1.0)
          image_alpha = torch.clamp(image_pkg["alpha"], 0.0, 1.0)
          image_gs = torch.clamp(
              renderFunc(
                  viewpoint,
                  scene.gaussians,
                  *renderArgs,
                  render_ellipsoid=True,
              )["render"], 0.0, 1.0)

          image_a1 = torch.clamp(
              renderFunc(
                  viewpoint,
                  scene.gaussians,
                  *renderArgs,
                  opacity_modifier=1.0,
              )["render"], 0.0, 1.0)

          gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
          if viewpoint.original_mask_sparse is not None:
            gt_mask = viewpoint.original_mask_sparse.to("cuda").to_dense()

            # print("gt_mask shape: ", gt_mask.shape)
            # show only the first 3 masks
            gt_mask = torch.cat([gt_mask[0:1], gt_mask[1:2], gt_mask[2:3]],
                                dim=0)
          else:
            gt_mask = None
          # print("gt_mask shape: ", gt_mask.shape)
          if tb_writer and (idx < 5):
            tb_writer.add_images(config['name'] +
                                 "_view_{}/render".format(viewpoint.image_name),
                                 image[None],
                                 global_step=iteration)
            tb_writer.add_images(
                config['name'] +
                "_view_{}/render_depth".format(viewpoint.image_name),
                image_depth[None],
                global_step=iteration)
            tb_writer.add_images(
                config['name'] +
                "_view_{}/render_sem".format(viewpoint.image_name),
                image_sem[None],
                global_step=iteration)
            tb_writer.add_images(
                config['name'] +
                "_view_{}/render_alpha".format(viewpoint.image_name),
                image_alpha[None],
                global_step=iteration)
            tb_writer.add_images(
                config['name'] +
                "_view_{}/render_gs".format(viewpoint.image_name),
                image_gs[None],
                global_step=iteration)

            tb_writer.add_images(
                config['name'] +
                "_view_{}/render_a1".format(viewpoint.image_name),
                image_a1[None],
                global_step=iteration)

            if iteration == testing_iterations[0]:
              tb_writer.add_images(
                  config['name'] +
                  "_view_{}/ground_truth".format(viewpoint.image_name),
                  gt_image[None],
                  global_step=iteration)
              if len(gt_mask.shape) == 3:
                # print("type {}", type(gt_mask))
                # print("gt_mask shape: ", gt_mask.shape)
                tb_writer.add_images(config['name'] +
                                     "_view_{}/ground_truth_mask".format(
                                         viewpoint.mask_sparse_name),
                                     gt_mask[None],
                                     global_step=iteration)
          l1_test += l1_loss(image, gt_image).mean().double()
          psnr_test += psnr(image, gt_image).mean().double()
        psnr_test /= len(config['cameras'])
        l1_test /= len(config['cameras'])
        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
            iteration, config['name'], l1_test, psnr_test))
        if tb_writer:
          tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss',
                               l1_test, iteration)
          tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr',
                               psnr_test, iteration)

    if tb_writer:
      tb_writer.add_histogram("scene/opacity_histogram",
                              scene.gaussians.get_opacity, iteration)
      tb_writer.add_histogram(
          "scene/radii_histogram",
          scene.gaussians._anchor_radii.view(-1, 1).contiguous(), iteration)
      tb_writer.add_histogram("scene/anchor_children_num_histogram",
                              scene.gaussians.get_num_children_per_anchor,
                              iteration)
      tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0],
                           iteration)
    torch.cuda.empty_cache()


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Training script parameters")
  lp = ModelParams(parser)
  op = OptimizationParams(parser)
  pp = PipelineParams(parser)
  parser.add_argument('--ip', type=str, default="127.0.0.1")
  parser.add_argument('--port', type=int, default=8081)
  parser.add_argument('--debug_from', type=int, default=-1)
  parser.add_argument('--detect_anomaly', action='store_true', default=False)
  parser.add_argument("--test_iterations",
                      nargs="+",
                      type=int,
                      default=[15_000, 20_000, 30_000, 35_000])
  parser.add_argument("--save_iterations",
                      nargs="+",
                      type=int,
                      default=[30_000, 35_000])
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--checkpoint_iterations",
                      nargs="+",
                      type=int,
                      default=[])
  parser.add_argument("--start_checkpoint", type=str, default=None)
  parser.add_argument("--load_iteration", type=int, default=None)
  # parser.add_argument('--use_free_scale', action='store_true', default=False)
  args = parser.parse_args(sys.argv[1:])
  args.save_iterations.append(args.iterations)

  print("Optimizing " + args.model_path)

  # Initialize system state (RNG)
  safe_state(args.quiet)

  # Start GUI server, configure and run training
  network_gui.init(args.ip, args.port)
  torch.autograd.set_detect_anomaly(args.detect_anomaly)

  training(lp.extract(args), op.extract(args), pp.extract(args),
           args.test_iterations, args.save_iterations,
           args.checkpoint_iterations, args.start_checkpoint,
           args.load_iteration, args.debug_from)
  # All done
  print("\nTraining complete.")
