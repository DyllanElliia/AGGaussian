import os
import sys
import torch
import torchvision
import clip
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange, repeat
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

try:
  from sam2.build_sam import build_sam2
  from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except:
  print("fail to import sam2.")

try:
  from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except:
  print("fail to import sam1.")

from utils.clip_utils import load_clip, get_features_from_image_and_masks
# sys.path.append("..")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.log_utils import *
from utils.math_utils import to_sparse, from_sparse


def set_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def replace_last_dir_segment(path: str, seg: str, newseg: str) -> str:
  p = Path(path)
  parts = list(p.parts)
  # 从右往左找目录段（跳过最后的文件名）
  for i in range(len(parts) - 2, -1, -1):
    if parts[i] == seg:
      parts[i] = newseg
      return str(Path(*parts))
  return str(p)


def show_anns(anns, borders=True):
  if len(anns) == 0:
    return
  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)

  img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                 sorted_anns[0]['segmentation'].shape[1], 4))
  img[:, :, 3] = 0
  for ann in sorted_anns:
    m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3), [0.5]])
    img[m] = color_mask
    if borders:
      import cv2
      contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
      # Try to smooth contours
      contours = [
          cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
          for contour in contours
      ]
      cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

  ax.imshow(img)


def show_masks(masks, borders=True):
  """
  Input:
      masks: torch.Tensor, shape (num_masks, H, W)
      borders: bool, whether to show the borders of the masks
  """
  n, H, W = masks.shape
  if n == 0:
    return
  area = torch.sum(masks, dim=(1, 2))
  sorted_idx = torch.argsort(area, descending=True)
  sorted_masks = masks[sorted_idx].cpu().numpy()
  sorted_area = area[sorted_idx].cpu().numpy()

  ax = plt.gca()
  ax.set_autoscale_on(False)

  img = np.ones((H, W, 4))
  img[:, :, 3] = 0

  for idx, (m, area) in enumerate(zip(sorted_masks, sorted_area)):
    if area == 0:
      continue
    # m = ann['segmentation']
    color_mask = np.concatenate([np.random.random(3), [0.5]])
    img[m] = color_mask
    if borders:
      import cv2
      contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
      # Try to smooth contours
      contours = [
          cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
          for contour in contours
      ]
      cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

  ax.imshow(img)


def plot_image(image, masks, savePath=None):
  plt.figure(figsize=(20, 20))
  plt.imshow(image * (0.2 / 255.))
  if isinstance(masks, torch.Tensor):
    show_masks(masks)
  elif isinstance(masks, list):
    show_anns(masks)
  else:
    raise ValueError("masks should be either torch.Tensor or list")
  plt.axis('off')
  if savePath:
    plt.savefig(savePath)
  else:
    plt.show()
  plt.close()


def get_seg_img(mask, image):
  """
  Get the segmented image from the mask
  
  Input:
  mask: dict, mask information
  image: np.array, image
  """
  image = image.copy()
  image[mask['segmentation'] == 0] = np.array([0, 0, 0], dtype=np.uint8)
  x, y, w, h = np.int32(mask['bbox'])
  seg_img = image[y:y + h, x:x + w, ...]
  return seg_img


def filter(keep: torch.Tensor, masks_result) -> None:
  keep = keep.int().cpu().numpy()
  result_keep = []
  for i, m in enumerate(masks_result):
    if i in keep:
      result_keep.append(m)
  return result_keep


def mask_nms(masks,
             scores,
             iou_thr=0.7,
             score_thr=0.1,
             inner_thr=0.2,
             **kwargs):
  """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """
  scores, idx = scores.sort(0, descending=True)
  num_masks = idx.shape[0]

  masks_ord = masks[idx.view(-1), :]
  masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

  iou_matrix = torch.zeros((num_masks,) * 2,
                           dtype=torch.float,
                           device=masks.device)
  inner_iou_matrix = torch.zeros((num_masks,) * 2,
                                 dtype=torch.float,
                                 device=masks.device)
  for i in range(num_masks):
    for j in range(i, num_masks):
      intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]),
                               dtype=torch.float)
      union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]),
                        dtype=torch.float)
      iou = intersection / union
      iou_matrix[i, j] = iou
      # select mask pairs that may have a severe internal relationship
      if intersection / masks_area[i] < 0.5 and intersection / masks_area[
          j] >= 0.85:
        inner_iou = 1 - (intersection / masks_area[j]) * (intersection /
                                                          masks_area[i])
        inner_iou_matrix[i, j] = inner_iou
      if intersection / masks_area[i] >= 0.85 and intersection / masks_area[
          j] < 0.5:
        inner_iou = 1 - (intersection / masks_area[j]) * (intersection /
                                                          masks_area[i])
        inner_iou_matrix[j, i] = inner_iou

  iou_matrix.triu_(diagonal=1)
  iou_max, _ = iou_matrix.max(dim=0)
  inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
  inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
  inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
  inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

  keep = iou_max <= iou_thr
  keep_conf = scores > score_thr
  keep_inner_u = inner_iou_max_u <= 1 - inner_thr
  keep_inner_l = inner_iou_max_l <= 1 - inner_thr

  # If there are no masks with scores above threshold, the top 3 masks are selected
  if keep_conf.sum() == 0:
    index = scores.topk(3).indices
    keep_conf[index, 0] = True
  if keep_inner_u.sum() == 0:
    index = scores.topk(3).indices
    keep_inner_u[index, 0] = True
  if keep_inner_l.sum() == 0:
    index = scores.topk(3).indices
    keep_inner_l[index, 0] = True
  keep *= keep_conf
  keep *= keep_inner_u
  keep *= keep_inner_l

  selected_idx = idx[keep]
  return selected_idx


def get_sam2(logger, extract_level, downscale_factor, point_per_batch, device):

  sam2_checkpoint = "./third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
  model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
  logger.info("Loading SAM2 model from: {}".format(sam2_checkpoint))
  sam2 = build_sam2(model_cfg,
                    sam2_checkpoint,
                    device=device,
                    apply_postprocessing=False)

  logger.info("Mask Generator initialized")
  downscale_factor = downscale_factor
  logger.info("Downscale Factor: {}".format(downscale_factor))
  logger.info("Points Per Batch: {}".format(point_per_batch))
  # self.mask_generator = SAM2AutomaticMaskGenerator(
  #     model=self.sam2,
  #     points_per_side=64,
  #     points_per_batch=point_per_batch,
  #     pred_iou_thresh=0.80,
  #     stability_score_thresh=0.93,
  #     stability_score_offset=0.7,
  #     crop_n_layers=1,
  #     box_nms_thresh=0.7,
  #     crop_n_points_downscale_factor=self.downscale_factor,
  #     min_mask_region_area=25.0,
  #     use_m2m=True,
  # )

  if extract_level == "detail":
    logger.info(f"Extracting {extract_level} level masks")
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=point_per_batch,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.7,
        crop_n_layers=1,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=downscale_factor,
        min_mask_region_area=50,
        use_m2m=True,
    )
  elif extract_level == "coarse":
    logger.info(f"Extracting {extract_level} level masks")
    # mask_generator = SAM2AutomaticMaskGenerator(
    #     model=sam2,
    #     points_per_side=32,
    #     points_per_batch=point_per_batch,
    #     pred_iou_thresh=0.88,
    #     stability_score_thresh=0.95,
    #     stability_score_offset=0.7,
    #     crop_n_layers=0,
    #     box_nms_thresh=0.7,
    #     crop_n_points_downscale_factor=1,
    #     min_mask_region_area=50,
    #     use_m2m=False,
    # )
    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_side=32,
        points_per_batch=point_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
        use_m2m=False,
    )
  elif extract_level == "default":
    logger.info(f"Extracting {extract_level} level masks")
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2)
  return mask_generator


def get_sam(logger, extract_level, downscale_factor, point_per_batch, device):

  sam_checkpoint = "./third_party/sam/checkpoints/sam_vit_h_4b8939.pth"
  model_cfg = "vit_h"
  logger.info("Loading SAM model from: {}".format(sam_checkpoint))
  sam = sam_model_registry[model_cfg](checkpoint=sam_checkpoint).to(device)

  logger.info("Mask Generator initialized")
  downscale_factor = downscale_factor
  logger.info("Downscale Factor: {}".format(downscale_factor))
  logger.info("Points Per Batch: {}".format(point_per_batch))
  if extract_level == "detail":
    logger.info(f"Extracting {extract_level} level masks")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=point_per_batch,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=50,
    )
  elif extract_level == "coarse":
    logger.info(f"Extracting {extract_level} level masks")
    # mask_generator = SamAutomaticMaskGenerator(
    #     model=sam,
    #     points_per_side=12,
    #     pred_iou_thresh=0.98,
    #     # stability_score_thresh=0.95,
    #     min_mask_region_area=100,
    # )
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=point_per_batch,
        pred_iou_thresh=0.88,
        box_nms_thresh=0.7,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
  else:
    logger.info(f"Extracting {extract_level} level masks")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_batch=point_per_batch,
    )
  return mask_generator


class mask_extractor:

  def __init__(self,
               device: torch.device = torch.device("cuda"),
               downscale_factor: int = 1,
               point_per_batch: int = 128,
               extract_level="coarse",
               use_sam2=False):
    self.device = device
    # self.logger = get_logger()
    self.logger = get_loguru_logger()
    self.logger.info("Mask Extractor initialized")
    self.logger.info("device: {}".format(device))
    assert extract_level in ["detail", "coarse", "default"]

    if use_sam2:
      if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
          torch.backends.cuda.matmul.allow_tf32 = True
          torch.backends.cudnn.allow_tf32 = True
      elif device.type == "mps":
        self.logger.info(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
      self.logger.info("Using SAM2")
      self.mask_generator = get_sam2(self.logger, extract_level,
                                     downscale_factor, point_per_batch,
                                     self.device)
    else:
      self.logger.info("Using SAM1")
      self.mask_generator = get_sam(self.logger, extract_level,
                                    downscale_factor, point_per_batch,
                                    self.device)
    self.logger.info("Mask Extractor Done")

  def _load_image(self, image_path, image_mode="RGBA"):
    image = Image.open(image_path)
    image = np.array(image.convert(image_mode))
    return image

  def _walk_root_dir(self, root_dir):
    file_paths = []
    for root, dirs, files in os.walk(root_dir):
      for file in files:
        filetype = file.split(".")[-1]
        if filetype in ["jpg", "JPG", "png"]:
          # yield os.path.join(root, file)
          file_paths.append(os.path.join(root, file))
    return sorted(file_paths)

  def load_images(self, root, image_mode="RGBA"):
    images = []
    image_folder = root
    image_paths = self._walk_root_dir(image_folder)
    # print(image_paths)
    # image_paths = image_paths[:2]  # for testing
    max_img_num = 8000
    if len(image_paths) > max_img_num:
      random_idx = np.random.permutation(len(image_paths))
      image_paths = [image_paths[i] for i in random_idx[:max_img_num]]
      # sort again
      image_paths = sorted(image_paths)
      self.logger.warning(
          "Too many images, randomly select {} images.".format(max_img_num))
    self.logger.info("Found {} images in {}".format(len(image_paths), root))

    for fp in tqdm(image_paths, desc="Loading Images"):
      image = self._load_image(fp)  # (H, W, C)
      # print(image.shape,
      #       image[..., 3].sum() / 255.0 / image.shape[0] / image.shape[1])
      images.append(image)
      # break
    return images, image_paths

  def _gen_masks(self, image):
    if image.shape[2] == 4:
      return self.mask_generator.generate(image[..., :3])
    else:
      return self.mask_generator.generate(image)

  # def gen_masks(self, images):
  #   masks = []
  #   for image in tqdm(images, desc="Generating Masks"):
  #     masks.append(self._gen_masks(image))
  #   torch.cuda.empty_cache()
  #   return masks

  def _process_mask(self, mask, alpha=None):
    mask_l = []
    iou_l = []
    stability_l = []

    alpha_mask = None
    if alpha is not None:
      alpha_mask = torch.from_numpy(alpha < 0.1).to(self.device).bool()

    for m in mask:
      mask_l.append(torch.from_numpy(m['segmentation']).to(self.device))

      iou_l.append(torch.tensor(m['predicted_iou']).to(self.device))

      stability_l.append(torch.tensor(m['stability_score']).to(self.device))

    mask = torch.stack(mask_l, dim=0)
    iou = torch.stack(iou_l, dim=0)
    stability = torch.stack(stability_l, dim=0)

    scores = stability * iou
    keep = mask_nms(mask, scores, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    mask = mask[keep].bool()
    if alpha_mask is not None:
      # print("Applying alpha mask")
      mask = mask & (~alpha_mask)
    # self.logger.error("Mask: {}".format(keep))
    # filter out the masks with area less than 25
    mask = mask[torch.sum(mask, dim=(1, 2)) > 25]
    return mask.cpu()

  def process_masks_to_torch(self, masks, images=None):
    masks_list = []

    for idx, mask in tqdm(enumerate(masks), desc="Processing Masks"):
      masks_list.append(
          self._process_mask(
              mask, alpha=images[idx][..., 3] if images is not None else None))

    torch.cuda.empty_cache()
    return masks_list

  def save_masks(self, masks, path_lists):
    for mask, path in tqdm(zip(masks, path_lists), desc="Saving Masks"):
      torch.save(to_sparse(mask), path)

  @torch.no_grad()
  def run(
      self,
      root,
      images_folder="images",
      image_mode="RGBA",
      masks_folder="masks",
      masks_view_folder="masks_view",
      extra_Mask=False,
      extra_clip=False,
      clip_features_folder="clip_features",
  ):
    self.logger.info("Extracting Masks from: {}".format(root))
    images, image_paths = self.load_images(os.path.join(root, images_folder),
                                           image_mode)

    image_file_type = image_paths[0].split(".")[-1]
    mask_paths = [
        replace_last_dir_segment(path, images_folder,
                                 masks_folder).replace(image_file_type, "pt")
        for path in image_paths
    ]
    if not os.path.exists(os.path.join(root, masks_folder)):
      self.logger.info("Creating Folder: {}".format(
          os.path.join(root, masks_folder)))
      os.makedirs(os.path.join(root, masks_folder))
    # visualize masks
    self.logger.info("Saving Masks to: {}".format(
        os.path.join(root, masks_folder)))
    if not os.path.exists(os.path.join(root, masks_view_folder)):
      os.makedirs(os.path.join(root, masks_view_folder))
    masks_view_path = [
        replace_last_dir_segment(path, images_folder, masks_view_folder)
        for path in image_paths
    ]
    # for i in tqdm(range(len(masks)), desc="Visualizing Masks"):
    #   plot_image(images[i], masks[i], savePath=masks_view_path[i])
    if extra_Mask:
      for image, image_path, mask_path, mask_view_path in tqdm(
          zip(images, image_paths, mask_paths, masks_view_path),
          desc="Generating Masks"):

        mask = self._gen_masks(image)
        # plot_image(image, mask, savePath=mask_view_path)
        mask_torch = self._process_mask(
            mask, alpha=image[..., 3] if image.shape[2] == 4 else None)
        if mask_torch.shape[1:] != image.shape[:2]:
          self.logger.warning(
              f"Shape mismatch: {mask_torch.shape[1:]} vs {image.shape[:2]}")
          mask_torch = torch.nn.functional.interpolate(
              mask_torch.unsqueeze(0).float(),
              size=(image.shape[0], image.shape[1]),
              mode='nearest')[0].bool()
        plot_image(image[..., :3], mask_torch, savePath=mask_view_path)
        # self.logger.debug("Mask shape: {}".format(mask_torch.shape))
        torch.save(to_sparse(mask_torch), mask_path)
        torch.cuda.empty_cache()
      # torch.cuda.empty_cache()
      del images
    # return
    if extra_clip:
      self.logger.info("Extracting CLIP features")
      self.logger.info("Delete sam2 model to save memory")
      del self.mask_generator
      # del self.sam2
      torch.cuda.empty_cache()
      if not os.path.exists(os.path.join(root, clip_features_folder)):
        self.logger.info("Creating Folder: {}".format(
            os.path.join(root, clip_features_folder)))
        os.makedirs(os.path.join(root, clip_features_folder))
      clip_model = load_clip(self.device)
      for image_path, mask_path in tqdm(zip(image_paths, mask_paths),
                                        desc="Extracting CLIP features"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = from_sparse(torch.load(mask_path,
                                       map_location=self.device)).float()
        # self.logger.debug("Masks shape: {}".format(masks.shape))
        features = get_features_from_image_and_masks(clip_model,
                                                     image,
                                                     masks,
                                                     background=0.)
        # self.logger.info("Features shape: {}".format(features.shape))
        torch.save(
            features.half(),
            replace_last_dir_segment(mask_path, masks_folder,
                                     clip_features_folder))

    # self.save_masks(masks_torch, mask_paths)
    self.logger.info("Done")


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--root", type=str, default="data")
  parser.add_argument("--images_folder", type=str, default="images")
  parser.add_argument("--image_mode", type=str, default="RGBA")
  parser.add_argument("--masks_folder", type=str, default="masks")
  parser.add_argument("--masks_view_folder", type=str, default="masks_view")
  parser.add_argument("--clip_features_folder",
                      type=str,
                      default="clip_features")
  parser.add_argument("--point_per_batch", type=int, default=256)
  parser.add_argument("--extract_level", type=str, default="coarse")
  parser.add_argument("--use_sam2", action="store_true")
  args = parser.parse_args()
  set_seed(114514)

  mask_extractor(point_per_batch=args.point_per_batch,
                 extract_level=args.extract_level,
                 use_sam2=args.use_sam2).run(args.root, args.images_folder,
                                             args.image_mode, args.masks_folder,
                                             args.masks_view_folder, True, True,
                                             args.clip_features_folder)
