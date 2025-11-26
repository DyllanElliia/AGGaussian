import os
import numpy as np
from PIL import Image
import cv2
from argparse import ArgumentParser


def load_image_as_binary(image_path, is_png=False, threshold=10):
  image = Image.open(image_path)
  if is_png:
    image = image.convert('L')
  image_array = np.array(image)
  binary_image = (image_array > threshold).astype(int)
  return binary_image


# def align_masks(mask1, mask2):
#   print(f"Aligning masks: {mask1.shape} and {mask2.shape}")
#   min_h = min(mask1.shape[0], mask2.shape[0])
#   min_w = min(mask1.shape[1], mask2.shape[1])
#   return mask1[:min_h, :min_w], mask2[:min_h, :min_w]
def align_masks(mask1, mask2):
  tgt_h = min(mask1.shape[0], mask2.shape[0])
  tgt_w = min(mask1.shape[1], mask2.shape[1])

  def _resize_nearest(img, h, w):
    if (img.shape[0], img.shape[1]) == (h, w):
      return img
    # cv2.resize 的 size 是 (w, h)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

  aligned1 = _resize_nearest(mask1, tgt_h, tgt_w)
  aligned2 = _resize_nearest(mask2, tgt_h, tgt_w)
  return aligned1, aligned2


def resize_mask(mask, target_shape):
  """Resize the mask to the target shape."""
  return np.array(
      Image.fromarray(mask).resize((target_shape[1], target_shape[0]),
                                   resample=Image.NEAREST))


def calculate_iou(mask1, mask2):
  # if mask1.shape != mask2.shape:
  #   mask1, mask2 = align_masks(mask1, mask2)
  intersection = np.logical_and(mask1, mask2).sum()
  union = np.logical_or(mask1, mask2).sum()
  if union == 0:
    return 0
  return intersection / union


# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
  """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
  h, w = mask.shape
  img_diag = np.sqrt(h**2 + w**2)
  dilation = int(round(dilation_ratio * img_diag))
  if dilation < 1:
    dilation = 1
  # Pad image so mask truncated by the image border is also considered as boundary.
  new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
  kernel = np.ones((3, 3), dtype=np.uint8)
  new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
  mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
  # G_d intersects G in the paper.
  return mask - mask_erode


def boundary_iou(gt, dt, dilation_ratio=0.02):
  """
    Compute boundary iou between two binary masks.
    :param gt (numpy array, uint8): binary mask
    :param dt (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary iou (float)
    """
  dt = (dt * 255 > 128).astype('uint8')
  gt = (gt * 255 > 128).astype('uint8')

  gt_boundary = mask_to_boundary(gt, dilation_ratio)
  dt_boundary = mask_to_boundary(dt, dilation_ratio)
  # print(f"gt_boundary: {gt_boundary.mean()}")
  # print(f"dt_boundary: {dt_boundary.mean()}")
  intersection = ((gt_boundary * dt_boundary) > 0).sum()
  # print(f"intersection: {intersection}")
  union = ((gt_boundary + dt_boundary) > 0).sum()
  # print(f"union: {union}")
  boundary_iou = intersection / union
  return boundary_iou


def evalute(gt_base, pred_base, scene_name, gt_is_png=False):
  scene_gt_frames = {
      "waldo_kitchen": [
          "frame_00053", "frame_00066", "frame_00089", "frame_00140",
          "frame_00154"
      ],
      "ramen": [
          "frame_00006", "frame_00024", "frame_00060", "frame_00065",
          "frame_00081", "frame_00119", "frame_00128"
      ],
      "figurines": ["frame_00041", "frame_00105", "frame_00152", "frame_00195"],
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
  frame_names = scene_gt_frames[scene_name]

  ious = []
  biou_scores = []
  for frame in frame_names:
    print("frame:", frame)
    gt_floder = os.path.join(gt_base, frame)
    if gt_is_png:
      file_names = [f for f in os.listdir(gt_floder) if f.endswith('.png')]
    else:
      file_names = [f for f in os.listdir(gt_floder) if f.endswith('.jpg')]
    for file_name in file_names:
      base_name = os.path.splitext(file_name)[0]
      gt_obj_path = os.path.join(gt_floder, file_name)
      pred_obj_path = os.path.join(pred_base, frame + "_" + base_name + '.png')
      if not os.path.exists(pred_obj_path):
        print(f"Missing pred file for {file_name}, skipping...")
        print(f"path: {pred_obj_path}")
        print(f"IoU for {file_name}: 0")
        ious.append(0.0)
        biou_scores.append(0.0)
        continue
      mask_gt = load_image_as_binary(gt_obj_path)
      mask_pred = load_image_as_binary(pred_obj_path, is_png=True)
      if mask_gt.shape != mask_pred.shape:
        mask_gt, mask_pred = align_masks(mask_gt, mask_pred)
      iou = calculate_iou(mask_gt, mask_pred)
      print("shape:", mask_gt.shape, mask_pred.shape)
      biou = boundary_iou(mask_gt, mask_pred)
      ious.append(iou)
      biou_scores.append(biou)
      print(
          f"IoU and BIoU for {file_name} and {base_name + '.png'}: {iou:.4f} and {biou:.4f}"
      )

  # Acc.
  total_count = len(ious)
  count_iou_025 = (np.array(ious) > 0.25).sum()
  count_iou_05 = (np.array(ious) > 0.5).sum()

  # mIoU
  average_iou = np.mean(ious)
  print(f"Average IoU: {average_iou:.4f}")
  print(f"Acc@0.25: {count_iou_025/total_count:.4f}")
  print(f"Acc@0.5: {count_iou_05/total_count:.4f}")
  # print(f"Acc@0.25 count: {count_iou_025}")
  # print(f"Total count: {total_count}")
  # print(f"my test: {42/56}")

  # Biou
  average_biou = np.mean(biou_scores)
  print(f"Average Biou: {average_biou:.4f}")


if __name__ == "__main__":
  parser = ArgumentParser("Compute LeRF IoU")
  parser.add_argument(
      "--scene_name",
      type=str,
      # choices=["waldo_kitchen", "ramen", "figurines", "teatime"],
      help=
      "Specify the scene_name from: figurines, teatime, ramen, waldo_kitchen")
  args = parser.parse_args()
  if not args.scene_name:
    parser.error(
        "The --scene_name argument is required and must be one of: waldo_kitchen, ramen, figurines, teatime"
    )

  # TODO: change
  path_gt = f"./datasets/label/{args.scene_name}/gt"
  is_png = False
  path_gt = f"./datasets/label_360/{args.scene_name}/gt"
  is_png = True
  # renders_cluster_silhouette is the predicted mask
  path_pred = f"./outputs/lerf/{args.scene_name}/text2obj/ours_35000"
  path_pred = f"./outputs/360_v2/{args.scene_name}/text2obj/ours_35000"

  evalute(path_gt, path_pred, args.scene_name, is_png)
