import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import shutil
import matplotlib.pyplot as plt
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.log_utils import *

from utils.math_utils import feature_map_to_rgb
from scripts.utils.clip_utils import *


def render_set(model_path,
               name,
               iteration,
               views,
               gaussians,
               pipeline,
               background,
               scene_texts,
               clip_model,
               candidate_frames,
               top_k=1):

  mask_path = os.path.join(model_path, name, "ours_{}".format(iteration))

  makedirs(mask_path, exist_ok=True)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  sa_clip = gaussians.LangHelper.sa_lang
  sa_clip /= sa_clip.norm(dim=-1, keepdim=True) + 1e-6

  sa_unmatch_mask = sa_clip.norm(-1)
  sa_unmatch_mask = sa_unmatch_mask > 10.0  # 1/1e-6 will be very large

  anchor_num_per_sg = gaussians.LangHelper.a2sa_idx.bincount().float()
  # remove_threshold = 2
  # sa_unquery = torch.where(anchor_num_per_sg < remove_threshold)[0]
  # # [TODO] select the top 1024 super anchors
  num = 1024
  if anchor_num_per_sg.shape[0] > num:
    sa_unquery = torch.topk(anchor_num_per_sg,
                            anchor_num_per_sg.shape[0] - num,
                            largest=False)[1]
  else:
    remove_threshold = 5
    sa_unquery = torch.where(anchor_num_per_sg < remove_threshold)[0]

  for text in scene_texts:
    text_clip = clip_model.tokenizer(text).to(device)
    text_clip = clip_model.model.encode_text(text_clip).half()  # (1, 512)
    text_clip = torch.nn.functional.normalize(text_clip, p=2, dim=-1)

    similarity = sa_clip @ text_clip.T
    similarity[sa_unquery] = -1.0
    similarity[sa_unmatch_mask] = -1.0
    similarity = (100 * similarity).softmax(dim=0)

    if False:
      # similarity-based anchor selection
      sa_on = similarity[:] > 0.2
    else:
      tao = 0.1
      # topk-based anchor selection
      value, indices = similarity[:].topk(top_k, dim=0)  # (top_k, 1)
      if top_k > 1:
        # indices = indices.squeeze(1)
        ind = torch.max(anchor_num_per_sg[indices], dim=0)[1]
        print(indices, ind, anchor_num_per_sg[indices])
        indices = indices[ind]

      sa_on = torch.zeros_like(similarity, dtype=torch.bool)
      sa_on[indices] = True
      sa_on = sa_on & (similarity[:] > value.min() - tao)
      # sa_on = sa_on & ((similarity[:] > value.min() - tao) &
      #                  (similarity[:] < value.max()))

    init_anchor_idx = sa_on[gaussians.LangHelper.a2sa_idx].any(dim=1)
    _, selected_anchor = gaussians.GraphHelper.select_edges(
        init_anchor_idx,
        edge_weight_threshold=0.05,
        iter_num=10,
        edge_select_condition=lambda ew, threshold: ew < threshold,
        big_anchor_mask=None)
    in_color = torch.zeros_like(gaussians._anchor_xyz).float()
    in_color[selected_anchor] = 1.0
    in_color = in_color[gaussians._point2anchor].contiguous()

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
      if view.image_name not in candidate_frames:
        continue
      mask = render(
          view,
          gaussians,
          pipeline,
          background,
          override_color=in_color,
          # opacity_mask=selected_anchor[
          #     gaussians._point2anchor].float().unsqueeze(1),
      )["render"]
      mask = (mask > 0.5).float()
      if mask.sum() < 50:
        continue
      torchvision.utils.save_image(
          mask, os.path.join(mask_path, f"{view.image_name}_{text}.png"))


def render_sets(dataset: ModelParams,
                iteration: int,
                pipeline: PipelineParams,
                skip_train: bool,
                skip_test: bool,
                scene_name,
                top_k=1):
  logger = get_loguru_logger(os.path.join(args.model_path))
  mask_path = os.path.join(args.model_path, "text2obj",
                           "ours_{}".format(iteration))

  if os.path.exists(mask_path):
    # remove existing folder safely using shutil (cross-platform and avoids shell injection)
    try:
      shutil.rmtree(mask_path)
    except Exception as e:
      # fallback to os.system only if shutil fails for some reason
      print(f"Failed to remove {mask_path} with shutil: {e}")
      try:
        os.system(f'rm -rf "{mask_path}"')
      except Exception:
        pass
  with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree,
                              init_child_num=5,
                              use_free_scale=dataset.use_free_scale,
                              logger=logger)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.GraphHelper.eval()
    gaussians.LangHelper.init_mlp(512)
    gaussians.LangHelper.load(scene.model_path)
    gaussians.LangHelper.eval()

    def debug_edge_weights(edges):
      features = gaussians.segment_activation(gaussians._feature).detach()

      feature_dis = torch.norm(features[edges[:, 0]] - features[edges[:, 1]],
                               dim=1,
                               keepdim=True)  # (m, 1)
      return feature_dis

    gaussians.GraphHelper.build_graph(edge_dist_scale=4, link_isolated=False)

    gaussians.GraphHelper.edge_weights = gaussians.GraphHelper.compute_all_edge_weight(
        debug_edge_weights).squeeze(1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    clip_model = load_clip()

    scene_texts = {
        "waldo_kitchen": ['Stainless steel pots', 'dark cup', 'refrigerator', 'frog cup', 'pot', 'spatula', 'plate', \
                'spoon', 'toaster', 'ottolenghi', 'plastic ladle', 'sink', 'ketchup', 'cabinet', 'red cup', \
                'pour-over vessel', 'knife', 'yellow desk'],
        "ramen": ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles'],
        "figurines": ['jake', 'pirate hat', 'pikachu', 'rubber duck with hat', 'porcelain hand', \
                    'red apple', 'tesla door handle', 'waldo', 'bag', 'toy cat statue', 'miffy', \
                    'green apple', 'pumpkin', 'rubics cube', 'old camera', 'rubber duck with buoy', \
                    'red toy chair', 'pink ice cream', 'spatula', 'green toy chair', 'toy elephant'],
        "teatime": ['sheep', 'yellow pouf', 'stuffed bear', 'coffee mug', 'tea in a glass', 'apple',
                'coffee', 'hooves', 'bear nose', 'dall-e brand', 'plate', 'paper napkin', 'three cookies', \
                'bag of cookies'
                ],
        "counter": ['Jar of coconut oil', 'fruit oranges', 'onions', 'plants', 'blue Oven Gloves', 'Wood Rolling Pin',
                    'free range eggs box', 'stable bread', 'Garofalo pasta', 'Napolina Tomatoes', 'gold Ripple Baking Pan','black granite texture plat'],
        "room": ["windows",
                 "brown shoes",
                 "blue grey chair",
                 "deep dark green carpets",
                 "yellow wood floors",
                 "silver gray curtain",
                 "white wood door",
                 "yellow books",
                 "wood desk",
                 "black loud speakers",
                 "Family Portrait Print",
                 "wine glasses and bottles",
                 "piano keyboard",
                 "green Yucca plant"],
        "bicycle": ["green grass", "bicycle", "tire", "bench", "Asphalt ground", "Silver Oak Tree"],
        "bonsai": ["piano keyboard", "bicycle", "purple table cloth", "black stool", "plastic bonsai tree", "dark grey patterned carpet"],
        "garden": ["doors",
                   "elderflower",
                   "green plant",
                   "green grass",
                   "wood table",
                   "wood pot",
                   "windows",
                   "Hexagonal Stone ground",
                   "football",
                   "dried flowers"],
        "kitchen": ["LEGO Technic 856 Bulldozer", "Basket Weave Cloth", "wood chair", "Wood plat", "old pink striped cloth", "Red Oven Gloves"]
    }

    target_text = scene_texts[scene_name]
    scene_gt_frames = {
        "waldo_kitchen": [
            "frame_00053", "frame_00066", "frame_00089", "frame_00140",
            "frame_00154"
        ],
        "ramen": [
            "frame_00006", "frame_00024", "frame_00060", "frame_00065",
            "frame_00081", "frame_00119", "frame_00128"
        ],
        "figurines": [
            "frame_00041", "frame_00105", "frame_00152", "frame_00195"
        ],
        "teatime": [
            "frame_00002", "frame_00025", "frame_00043", "frame_00107",
            "frame_00129", "frame_00140"
        ],
        "counter": [
            "DSCF5857", "DSCF5865", "DSCF5875", "DSCF5885", "DSCF5895",
            "DSCF5905", "DSCF5915", "DSCF5925", "DSCF5935", "DSCF5945",
            "DSCF5955", "DSCF5965", "DSCF5975", "DSCF5985", "DSCF5995",
            "DSCF6005", "DSCF6015", "DSCF6025", "DSCF6035", "DSCF6045",
            "DSCF6055", "DSCF6065", "DSCF6075", "DSCF6085"
        ],
        "room": [
            "DSCF4670", "DSCF4680", "DSCF4690", "DSCF4700", "DSCF4721",
            "DSCF4730", "DSCF4759", "DSCF4770", "DSCF4780", "DSCF4795",
            "DSCF4800", "DSCF4816", "DSCF4854", "DSCF4867", "DSCF4880",
            "DSCF4908", "DSCF4920", "DSCF4925", "DSCF4958"
        ],
        "bicycle": [
            "_DSC8683", "_DSC8695", "_DSC8705", "_DSC8709", "_DSC8713",
            "_DSC8715", "_DSC8721", "_DSC8733", "_DSC8738", "_DSC8746",
            "_DSC8751", "_DSC8756", "_DSC8776", "_DSC8812", "_DSC8824",
            "_DSC8830", "_DSC8836", "_DSC8848", "_DSC8872"
        ],
        "bonsai": [
            "DSCF5565", "DSCF5575", "DSCF5585", "DSCF5595", "DSCF5605",
            "DSCF5615", "DSCF5625", "DSCF5635", "DSCF5645", "DSCF5655",
            "DSCF5660", "DSCF5675", "DSCF5685", "DSCF5695", "DSCF5710",
            "DSCF5720", "DSCF5730", "DSCF5740", "DSCF5745", "DSCF5755",
            "DSCF5765", "DSCF5775", "DSCF5785", "DSCF5790", "DSCF5800",
            "DSCF5810", "DSCF5824", "DSCF5838", "DSCF5850"
        ],
        "garden": [
            "DSC07956", "DSC07960", "DSC07970", "DSC07980", "DSC07990",
            "DSC08000", "DSC08010", "DSC08020", "DSC08030", "DSC08040",
            "DSC08050", "DSC08060", "DSC08070", "DSC08080", "DSC08090",
            "DSC08095", "DSC08120", "DSC08135"
        ],
        "kitchen": [
            "DSCF0660", "DSCF0670", "DSCF0680", "DSCF0690", "DSCF0700",
            "DSCF0710", "DSCF0720", "DSCF0730", "DSCF0740", "DSCF0750",
            "DSCF0760", "DSCF0775", "DSCF0785", "DSCF0795", "DSCF0805",
            "DSCF0815", "DSCF0822", "DSCF0825", "DSCF0835", "DSCF0845",
            "DSCF0855", "DSCF0865", "DSCF0880", "DSCF0890", "DSCF0900",
            "DSCF0910", "DSCF0920", "DSCF0930"
        ]
    }
    candidate_frames = scene_gt_frames[scene_name]

    if not skip_train:
      render_set(dataset.model_path,
                 "text2obj",
                 scene.loaded_iter,
                 scene.getTrainCameras(),
                 gaussians,
                 pipeline,
                 background,
                 target_text,
                 clip_model,
                 candidate_frames,
                 top_k=top_k)

    if not skip_test:
      render_set(dataset.model_path,
                 "text2obj",
                 scene.loaded_iter,
                 scene.getTestCameras(),
                 gaussians,
                 pipeline,
                 background,
                 target_text,
                 clip_model,
                 candidate_frames,
                 top_k=top_k)


if __name__ == "__main__":
  # Set up command line argument parser
  parser = ArgumentParser(description="Testing script parameters")
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument("--iteration", default=-1, type=int)
  parser.add_argument("--skip_train", action="store_true")
  parser.add_argument("--skip_test", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument(
      "--scene_name",
      type=str,
      choices=[
          "waldo_kitchen", "ramen", "figurines", "teatime", "counter", "room",
          "bicycle", "bonsai", "garden", "kitchen"
      ],
      help=
      "Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
  parser.add_argument("--top_k", default=1, type=int)
  args = get_combined_args(parser)
  print("Rendering " + args.model_path)

  if not args.scene_name:
    parser.error(
        "The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime"
    )

  # Initialize system state (RNG)
  safe_state(args.quiet)

  render_sets(model.extract(args), args.iteration, pipeline.extract(args),
              args.skip_train, args.skip_test, args.scene_name)
