import os
import torch
from torch import nn
import torchvision
import torch_scatter
import numpy as np
from plyfile import PlyData, PlyElement
from torch.cuda.amp import autocast
from functools import reduce
from einops import rearrange, repeat

from tqdm.auto import tqdm

from .. import GaussianModel
from .modules.mlp import *


class LangHelper_IoUmatch:

  def __init__(self, logger, gaussian_model: GaussianModel):
    self.logger = logger
    self.model = gaussian_model
    self.device = 'cuda:0'

    self.is_available = False
    self.clip_model = None

    self.model_path = "./"

  def need_train(self):
    return False

  def init_sa(self, sa_xyz, sa_f, a2sa_idx):
    self.sa_xyzf = torch.cat([sa_xyz, sa_f], dim=1)
    # self.sa_feat = self.sa_xyzf[:, 3:].clone()
    self.sa_feat = self.sa_xyzf.clone()
    # self.sa_f = sa_f
    self.a2sa_idx = a2sa_idx.clone()
    self.is_available = True

  def init_clip(self, clip_model):
    if not self.is_available:
      self.logger.error('Lang_helper is not available')
      return
    self.clip_model = clip_model

  def init_mlp(self, clip_f):
    return

  def init_neural_mapping(self, clip_f=512):
    return

  def save(self, root):
    torch.save(self.sa_feat, os.path.join(root, 'sa_feat.pth'))
    torch.save(self.a2sa_idx, os.path.join(root, 'a2sa_idx.pth'))
    torch.save(self.sa_lang, os.path.join(root, 'sa_lang.pth'))

  def load(self, root):
    self.sa_feat = torch.load(os.path.join(root, 'sa_feat.pth'))
    self.a2sa_idx = torch.load(os.path.join(root, 'a2sa_idx.pth'))
    self.sa_lang = torch.load(os.path.join(root, 'sa_lang.pth'))

  def eval(self):
    return

  def train(self):
    return

  def step(self, scaler=None):
    return

  def mapping_sa_feature_map(self, xyzf_map):
    """
    Input:
      xyzf_map: (3+3, H, W)
    Output:
      feature_map: (clip_f, H, W)
    """
    h, w = xyzf_map.shape[1:]
    xyzf_map = rearrange(xyzf_map, 'c h w -> (h w) c')
    return rearrange(xyzf_map, '(h w) c -> c h w', h=h, w=w), 0.0

  def get_anchor_super_feat(self):
    return self.sa_feat[self.a2sa_idx].contiguous()

  def get_gaussian_super_feat(self, use_transfer=True):
    # if use_transfer:
    #   return self.transfer(
    #       self.sa_feat)[self.a2sa_idx[self.model._point2anchor]].contiguous()
    # else:
    #   return self.sa_feat[:, :3][self.a2sa_idx[
    #       self.model._point2anchor]].contiguous()
    return self.sa_feat[self.a2sa_idx[self.model._point2anchor]].contiguous()

  def sa2anchor(self, sa_tensor):
    return sa_tensor[self.a2sa_idx].contiguous()

  def sa2gaussian(self, sa_tensor):
    return sa_tensor[self.a2sa_idx[self.model._point2anchor]].contiguous()

  def render_func(self, gs_color, render, cam_view, pipe, background):
    render_pkg = render(
        cam_view,
        self.model,
        pipe,
        background,
        override_color=gs_color,
        # override_semantics=gs_color,
        lang_mode=True,
    )
    return render_pkg["render"], render_pkg["feature"]

  def match(self, render, scene, pipe, background):
    with torch.no_grad():
      torch.cuda.empty_cache()
      sorted_train_cameras = sorted(scene.getTrainCameras(),
                                    key=lambda Camera: Camera.image_name)
      for view in sorted_train_cameras:
        view.original_mask_sparse = view.original_mask_sparse.cpu()
        view.clip_feat = view.clip_feat.cpu()
      torch.cuda.empty_cache()
      # compute the anchor number per super anchor
      anchor_num = self.a2sa_idx.bincount().float()
      # record the super anchor which has more than 4 anchors
      sa_idxs = torch.where(anchor_num > 4)[0]
      self.logger.info(
          f"Super anchor number: {sa_idxs.shape[0]} total: {self.sa_feat.shape[0]}"
      )

      # init match info
      match_info = torch.zeros(
          self.sa_feat.shape[0], len(sorted_train_cameras),
          3).half().to(self.device)  # (SA, V, <mask id, match score, is_match>)
      gs_color_to_dtype = self.model.get_xyz.dtype
      sa_colors = torch.zeros(self.sa_feat.shape[0],
                              3,
                              dtype=gs_color_to_dtype,
                              device=self.device)
      color_map = torch.tensor(
          [
              [1.0, 0.0, 0.0],  # r
              [0.0, 1.0, 0.0],  # g
              [0.0, 0.0, 1.0],  # b
          ],
          device=self.device).to(self.sa_feat.dtype)
      white_map = torch.tensor(
          [
              [0.0, 0.0, 0.0],  # r
              [0.0, 0.0, 0.0],  # g
              [0.0, 0.0, 0.0],  # b
          ],
          device=self.device).to(self.sa_feat.dtype)
      # -------------------------------------
      for i in tqdm(range(0, sa_idxs.shape[0], 3), desc="Matching"):
        # if i < 2670:
        #   continue
        i_end = min(i + 3, sa_idxs.shape[0] - 1)
        if i_end - i < 3:
          self.logger.debug(f"i_end {i_end}, i {i}")
        current_chunk = sa_idxs[i:i_end]

        current_num = i_end - i
        sa_colors[current_chunk] = color_map[:current_num]
        # self.logger.debug(f"sa_colors {sa_colors[current_chunk]}")

        gs_color = self.sa2gaussian(sa_colors).to(gs_color_to_dtype)
        # self.logger.debug(
        #     f"shape {gs_color.shape},gs_xyz {self.model.get_xyz.shape}")

        for view_id, view in enumerate(sorted_train_cameras):
          # render the super anchor
          iou_map, feature_map = self.render_func(gs_color, render, view, pipe,
                                                  background)
          mask_shape = view.original_mask_sparse.shape
          if iou_map.shape[1:] != mask_shape[1:]:
            iou_map = torch.nn.functional.interpolate(
                iou_map.unsqueeze(0),
                size=(mask_shape[1], mask_shape[2]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
            feature_map = torch.nn.functional.interpolate(
                feature_map.unsqueeze(0),
                size=(mask_shape[1], mask_shape[2]),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

          iou_map = (iou_map > 0.7).short()  # (n, H, W)
          if iou_map.sum() < 5:
            continue

          mask = view.original_mask_sparse.cuda().to_dense().bool()  # (m, H, W)

          # iou (n, m)
          iou_map = iou_map.bool()
          iou = self.calculate_iou(mask, iou_map)

          # feature distance
          # compute the mean feature of each mask (m, f)
          mask = mask.float()
          pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (m,1)
          # shape (m, f)
          mask_feat_mean = torch.einsum('fhw,mhw->mf', feature_map,
                                        mask) / pixel_per_mask

          debug_mask_feat_map = torch.einsum('mf,mhw->fhw', mask_feat_mean,
                                             mask)

          iou_map = iou_map.float()
          pixel_per_iou = iou_map.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)
          # replace the zero with 1 to avoid nan problem
          pixel_per_iou[pixel_per_iou == 0] = 1
          # shape (n, f)
          iou_feat_mean = torch.einsum('fhw,nhw->nf', feature_map,
                                       iou_map) / pixel_per_iou

          debug_iou_feat_map = torch.einsum('nf,nhw->fhw', iou_feat_mean,
                                            iou_map)

          l1_dis, _ = self.calculate_pairwise_distances(
              iou_feat_mean, mask_feat_mean,
              metric="l1")  # (n,f),(m,f) -> (n,m)

          scores = iou * (1 - l1_dis)
          if torch.isnan(scores).sum() > 0:
            self.logger.error(
                f"[{view_id}] nan in scores {scores.shape} {scores.max()} {scores.min()}"
            )
          if torch.isnan(iou).sum() > 0:
            self.logger.error(
                f"[{view_id}] nan in iou {iou.shape} {iou.max()} {iou.min()}")
          if torch.isnan(l1_dis).sum() > 0:
            self.logger.error(
                f"[{view_id}] nan in l1_dis {l1_dis.shape} {l1_dis.max()} {l1_dis.min()}"
            )

          # if view_id == 6:
          #   # if debug path not exist, create it
          #   root_path = "./debug"
          #   if not os.path.exists(root_path):
          #     self.logger.info(f"Create debug path {root_path}")
          #     os.makedirs(root_path)
          #   # save the image for debug
          #   torchvision.utils.save_image(
          #       0.5 * (feature_map * 0.5 + 0.5) + 0.5 * iou_map,
          #       os.path.join(root_path, f"i{i}iou.png"))
          #   torchvision.utils.save_image(
          #       debug_mask_feat_map * 0.5 + 0.5,
          #       os.path.join(root_path, f"i{i}mask_feat.png"))
          #   torchvision.utils.save_image(
          #       debug_iou_feat_map * 0.5 + 0.5,
          #       os.path.join(root_path, f"i{i}iou_feat.png"))
          # save the max score and idx to match_info
          max_score, max_ind = torch.max(scores, dim=-1)  # [n]
          # nan check
          if torch.isnan(max_score).sum() > 0:
            self.logger.error(f"[{view_id}] nan in max_score {max_score.shape}")
          if torch.isnan(max_ind).sum() > 0:
            self.logger.error(
                f"[{view_id}] nan in max_ind {max_ind.shape} {max_ind.max()} {max_ind.min()}"
            )
          is_matched = max_score > 0.3  # todo in OpenGaussian
          max_score[~is_matched] *= 0
          max_ind[~is_matched] *= 0
          match_info[current_chunk, view_id] = torch.stack(
              (max_ind, max_score, is_matched), dim=1).half()[:current_num]

        sa_colors[current_chunk] = white_map[:current_num]
        # self.logger.debug(f"sa_colors {sa_colors[current_chunk]}")
      torch.cuda.empty_cache()
      self.logger.info("Finish matching!")
      # save the match info
      torch.save(match_info, os.path.join(self.model_path, "match_info.pth"))
      del match_info
      # -------------------------------------
      torch.cuda.empty_cache()
      match_info = torch.load(os.path.join(self.model_path,
                                           "match_info.pth")).to(self.device)

      # get the match mask_id of all views per super anchor
      sa_per_view_matched_mask = match_info[:, :, 0].long()  # (SA, V)
      sa_per_view_match_score = match_info[:, :, 1].float()  # (SA, V)

      sa_per_view_is_matched = match_info[:, :, 2].float()  # (SA, V)

      match_info_sum = match_info.sum(
          dim=1)  # (SA, <matched_mask_id, matched_score, is_matched>)
      sa_occu_score = match_info_sum[:, 1]  # (SA) total score
      sa_per_view_match_score = torch.softmax(
          sa_per_view_match_score * 10, dim=1) * sa_occu_score.unsqueeze(1)
      sa_occu_count = match_info_sum[:,
                                     2]  # (SA) number of matches for each super anchor

      self.logger.info("Accumulated 2D Lang Feature to each super gaussian")
      per_sa_lang_sum = torch.zeros(self.sa_feat.shape[0], 512).to(self.device)
      # for v_id, view in enumerate(sorted_train_cameras):
      score_weight = 1.0
      is_match_weight = 0.1
      for v_id, view in tqdm(enumerate(sorted_train_cameras),
                             desc="Accumulate"):
        lang_feat = view.clip_feat.cuda()  # (m, l)

        single_view_leaf_feat = lang_feat[
            sa_per_view_matched_mask[:, v_id]].float() * (
                score_weight *
                sa_per_view_match_score[:, v_id].unsqueeze(1).float() +
                is_match_weight *
                sa_per_view_is_matched[:, v_id].unsqueeze(1).float())
        if torch.isnan(single_view_leaf_feat).sum() > 0:
          self.logger.error(
              f"[{v_id}] nan in single_view_leaf_feat {single_view_leaf_feat.shape}"
          )
        per_sa_lang_sum += single_view_leaf_feat
        torch.cuda.empty_cache()
      per_sa_lang_sum = per_sa_lang_sum / (
          score_weight * sa_occu_score.float() +
          is_match_weight * sa_occu_count.float() + 1e-6).unsqueeze(1)
      self.sa_lang = per_sa_lang_sum.half()
      self.logger.info("Finish Accumulation!")
    return

  def get_text_clip(self, text):
    """
    Input:
      text: string list
    Output:
      clip: (len(text), clip_f)
    """
    if self.clip_model is None:
      self.logger.error('Please init clip model!')
      return
    return self.clip_model.encode_text(text)

  def calculate_iou(self, masks1, masks2, base=None):
    """
    Calculate the Intersection over Union (IoU) between two sets of masks.
    Args:
        masks1: PyTorch tensor of shape [n, H, W], torch.int32.
        masks2: PyTorch tensor of shape [m, H, W], torch.int32.
    Returns:
        iou_matrix: PyTorch tensor of shape [m, n], containing IoU values.
    """
    # Ensure the masks are of type torch.int32
    if masks1.dtype != torch.bool:
      masks1 = masks1.to(torch.bool)
    if masks2.dtype != torch.bool:
      masks2 = masks2.to(torch.bool)

    # Expand masks to broadcastable shapes
    masks1_expanded = masks1.unsqueeze(0)  # [1, n, H, W]
    masks2_expanded = masks2.unsqueeze(1)  # [m, 1, H, W]

    # Compute intersection
    intersection = (masks1_expanded & masks2_expanded).float().sum(
        dim=(2, 3))  # [m, n]

    # Compute union
    if base == "former":
      union = (masks1_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    elif base == "later":
      union = (masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    else:
      union = (masks1_expanded |
               masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]

    # Compute IoU
    iou_matrix = intersection / union

    return iou_matrix

  def calculate_pairwise_distances(self, tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between every pair of vectors
    in two tensors of shape [m, 3] and [n, 3].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [m, 3].
        tensor2 (torch.Tensor): Another tensor of shape [n, 3].
    Returns:
        torch.Tensor: L1 distances of shape [m, n].
        torch.Tensor: L2 distances of shape [m, n].
    """
    # Reshape tensors to allow broadcasting
    # tensor1 shape becomes [m, 1, 3] and tensor2 shape becomes [1, n, 3]
    tensor1 = tensor1.unsqueeze(1)  # Now tensor1 is [m, 1, 3]
    tensor2 = tensor2.unsqueeze(0)  # Now tensor2 is [1, n, 3]

    # Compute the L1 distance
    if metric == "l1":
      return torch.abs(tensor1 - tensor2).sum(dim=2), None  # Result is [m, n]

    # Compute the L2 distance
    if metric == "l2":
      return None, torch.sqrt(
          (tensor1 - tensor2).pow(2).sum(dim=2))  # Result is [m, n]

    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=2)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))
    return l1_distances, l2_distances


# LangHelper = LangHelper_MLP
LangHelper = LangHelper_IoUmatch
