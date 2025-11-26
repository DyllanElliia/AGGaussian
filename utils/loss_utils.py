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
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from einops import rearrange, repeat

# ------------------------------------------------------------------------------------
# Image Loss


def l1_loss(network_output, gt):
  return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
  return ((network_output - gt)**2).mean()


def gaussian(window_size, sigma):
  gauss = torch.Tensor([
      exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
      for x in range(window_size)
  ])
  return gauss / gauss.sum()


def create_window(window_size, channel):
  _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
  _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
  window = Variable(
      _2D_window.expand(channel, 1, window_size, window_size).contiguous())
  return window


def ssim(img1, img2, window_size=11, size_average=True):
  channel = img1.size(-3)
  window = create_window(window_size, channel)

  if img1.is_cuda:
    window = window.cuda(img1.get_device())
  window = window.type_as(img1)

  return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
  mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
  mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

  mu1_sq = mu1.pow(2)
  mu2_sq = mu2.pow(2)
  mu1_mu2 = mu1 * mu2

  sigma1_sq = F.conv2d(
      img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
  sigma2_sq = F.conv2d(
      img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
  sigma12 = F.conv2d(
      img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

  C1 = 0.01**2
  C2 = 0.03**2

  ssim_map = ((2 * mu1_mu2 + C1) *
              (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                     (sigma1_sq + sigma2_sq + C2))

  if size_average:
    return ssim_map.mean()
  else:
    return ssim_map.mean(1).mean(1).mean(1)


# ------------------------------------------------------------------------------------

# mask loss


def sample_pos_neg_features(feat, mask, sample_pos_num=100, sample_neg_num=100):
  """
    Sample positive and negative features from the feature map based on the mask.

    Inputs:
    - feat: Feature map, shape (d, h, w)
    - mask: Binary mask, shape (n, h, w)
    - sample_pos_num: Number of positive samples to sample
    - sample_neg_num: Number of negative samples to sample

    Returns:
    - pos_feat: Tensor of positive features, shape (n, sample_pos_num, d)
    - neg_feat: Tensor of negative features, shape (n, sample_neg_num, d)
    """
  d, h, w = feat.shape
  hw = h * w
  n = mask.shape[0]

  F = feat.view(d, hw).t()  # (hw, d)

  M = mask.view(n, hw)  # (n, hw)

  # get non-mask region

  R_nonMask = ~(M.sum(dim=0, keepdim=True) == 0)  # (1, hw)
  R_nonMask = R_nonMask.float()

  # define large negative value for negative samples
  large_negative = -1e9

  # generate random values
  rand_vals = torch.rand((n, hw), device=feat.device)  # (n, hw)

  # calculate scores for positive samples
  M_float = M.float()  # (n, hw)
  M_inv_float = 1.0 - M_float  # or use (~M).float()
  M_float = M_float * R_nonMask
  M_inv_float = M_inv_float * R_nonMask
  # M_float = M_float * R_nonMask
  scores_pos = M_float * rand_vals + M_inv_float * large_negative  # (n, hw)

  # get positive sample indices
  _, pos_indices = torch.topk(scores_pos, k=sample_pos_num,
                              dim=1)  # (n, sample_pos_num)

  # calculate scores for negative samples
  # M_inv_float = M_inv_float * R_nonMask
  scores_neg = M_inv_float * rand_vals + M_float * large_negative  # (n, hw)

  # get negative sample indices
  _, neg_indices = torch.topk(scores_neg, k=sample_neg_num,
                              dim=1)  # (n, sample_neg_num)

  # flatten indices
  pos_indices_flat = pos_indices.view(-1)  # (n * sample_pos_num)
  neg_indices_flat = neg_indices.view(-1)  # (n * sample_neg_num)

  # gather positive and negative samples
  pos_feat = F[pos_indices_flat].view(n, sample_pos_num,
                                      d)  # (n, sample_pos_num, d)
  neg_feat = F[neg_indices_flat].view(n, sample_neg_num,
                                      d)  # (n, sample_neg_num, d)

  return pos_feat, neg_feat


def mask_constrastive_loss(feat,
                           mask,
                           tau=1.0,
                           sample_num=1024,
                           alpha=1.0,
                           beta=1.0):
  """
  Input:
    feat: Feature map (d, h, w)
    mask: n bool Mask (n, h, w)
  Output:
    loss
  """
  pos_feat, neg_feat = sample_pos_neg_features(
      feat, mask, sample_pos_num=sample_num,
      sample_neg_num=sample_num)  # (n, sample_num, d)
  n, s, d = pos_feat.shape

  pos_feat_exp1 = pos_feat.unsqueeze(2)  # (n, sample_num, 1, d)
  pos_feat_exp2 = pos_feat.unsqueeze(1)  # (n, 1, sample_num, d)
  pos_dist = torch.norm(pos_feat_exp1 - pos_feat_exp2,
                        dim=-1)  # (n, sample_num, sample_num)

  pos_feat_exp = pos_feat.unsqueeze(2)  # (n, sample_num, 1, d)
  neg_feat_exp = neg_feat.unsqueeze(1)  # (n, 1, sample_num, d)
  neg_dist = torch.norm(pos_feat_exp - neg_feat_exp,
                        dim=-1)  # (n, sample_num, sample_num)

  eye = torch.eye(sample_num, device=pos_feat.device).bool()
  pos_mask = ~eye.unsqueeze(0).expand(n, sample_num,
                                      sample_num)  # (n, sample_num, sample_num)
  pos_dist = pos_dist[pos_mask].view(n, sample_num, sample_num - 1)

  pos_loss = pos_dist.mean()
  neg_loss = F.relu(tau - neg_dist).mean()
  return alpha * pos_loss + beta * neg_loss


def mask_constrastive_loss_2(feat,
                             mask,
                             tau=1.0,
                             sample_pos_num=1024,
                             sample_neg_num=1024,
                             alpha=1.0,
                             beta=1.0):
  """
  Input:
    feat: Feature map (d, h, w)
    mask: n bool Mask (n, h, w)
  Output:
    loss
  """
  pos_feat, neg_feat = sample_pos_neg_features(
      feat, mask, sample_pos_num=sample_pos_num, sample_neg_num=sample_neg_num
  )  # (n, sample_pos_num, d), (n, sample_neg_num, d)
  n, _, d = pos_feat.shape

  pos_feat_exp1 = pos_feat.unsqueeze(2)  # (n, sample_pos_num, 1, d)
  pos_feat_exp2 = pos_feat.unsqueeze(1)  # (n, 1, sample_pos_num, d)
  pos_dist = torch.norm(pos_feat_exp1 - pos_feat_exp2,
                        dim=-1)  # (n, sample_pos_num, sample_pos_num)

  pos_feat_exp = pos_feat.unsqueeze(2)  # (n, sample_pos_num, 1, d)
  neg_feat_exp = neg_feat.unsqueeze(1)  # (n, 1, sample_neg_num, d)
  neg_dist = torch.norm(pos_feat_exp - neg_feat_exp,
                        dim=-1)  # (n, sample_pos_num, sample_neg_num)

  eye = torch.eye(sample_pos_num, device=pos_feat.device).bool()
  pos_mask = ~eye.unsqueeze(0).expand(
      n, sample_pos_num, sample_pos_num)  # (n, sample_pos_num, sample_pos_num)
  pos_dist = pos_dist[pos_mask].view(n, sample_pos_num, sample_pos_num - 1)

  pos_loss = pos_dist.mean()
  neg_loss = F.relu(tau - neg_dist).mean()
  # neg_loss = (tau - neg_dist).mean()
  return alpha * pos_loss + beta * neg_loss


# ------------------------------------------------------------------------------------
# Mask Loss v2


def og_mask_constrastive_loss_org(feat,
                                  mask,
                                  alpha=1.0,
                                  beta=1.0,
                                  gamma=0.1,
                                  epsilon=0.1,
                                  mode='l2',
                                  feat_norm_switch=False):
  """
  Input:
    feat: Feature map (d, h, w)
    mask: n bool Mask (n, h, w)
    alpha: weight for the intra-mask smoothing loss
    beta: weight for the inter-mask constrastive loss
  Output:
    loss
  """
  mask = mask.float()
  N = mask.shape[0]
  if feat.shape[1:] != mask.shape[1:]:
    feat = F.interpolate(feat.unsqueeze(0),
                         size=mask.shape[1:],
                         mode='bilinear',
                         align_corners=False).squeeze(0)
  if feat_norm_switch:
    feat = feat / torch.clamp_min(torch.norm(feat, p=2, dim=0, keepdim=True),
                                  1e-6)

  pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)

  # feat_mean: (n, d) <- sum(feat(1, d, h, w) * mask(n, 1, h, w))/(pixel_per_mask(n,1))
  feat_mean = torch.einsum('dhw,nhw->nd', feat, mask) / pixel_per_mask

  # --------------------------------------
  assert mode in ['l2', 'cos']
  if mode == 'l2':
    # feat-f_mean: (n, d, hw) <- feat(1, d, h, w) - feat_mean(n, d, 1)
    dist = rearrange(feat, 'd h w -> 1 d (h w)') - feat_mean.unsqueeze(
        2)  # (n, d, hw)
    mask_dist = (dist * rearrange(mask, 'n h w -> n 1 (h w)')).norm(
        p=2, dim=1)  # (n, hw)
  if mode == 'cos':
    # feat-f_mean: (n, d, hw) <- cos<feat(1, d, h*w), feat_mean(n, d, 1)>
    dist = -rearrange(feat, 'd h w -> 1 d (h w)') * feat_mean.unsqueeze(
        2)  # (n, d, hw)
    mask_dist = (dist * rearrange(mask, 'n h w -> n 1 (h w)')).sum(
        dim=1)  # (n, hw)

  intra_mask_smoothing_loss = mask_dist.mean()
  # mask_dist = dist * rearrange(mask, 'n h w -> n 1 (h w)')  # (n, d, hw)
  # intra_mask_smoothing_loss = torch.norm(mask_dist, p=2, dim=1).mean()
  # --------------------------------------
  # inter_mask_constrastive_loss
  if mode == 'l2':
    # feat_mean_dist: (n, n, d) <- feat_mean(1, n, d) - feat_mean(n, 1, d)
    feat_mean_dist = torch.norm(feat_mean.unsqueeze(1) - feat_mean.unsqueeze(0),
                                p=2,
                                dim=2)  # (n, n)
  if mode == 'cos':
    # feat_mean_dist: (n, n, d) <- cos<feat_mean(1, n, d), feat_mean(n, 1, d)>
    feat_mean_dist = -(feat_mean.unsqueeze(1) * feat_mean.unsqueeze(0)).sum(
        dim=2)  # (n, n)
  eye = ~torch.eye(feat_mean_dist.shape[0], device=feat.device).bool()  # (n, n)
  feat_mean_dist = feat_mean_dist[eye].view(
      feat_mean_dist.shape[0], feat_mean_dist.shape[0] - 1)  # (n, n-1)
  inverse_f_m_d = 1.0 / (feat_mean_dist + epsilon)

  # ReWeight it
  sorted_indices = inverse_f_m_d.argsort().argsort()
  loss_weight = (sorted_indices.float() /
                 (N - 1)) * (1.0 - 0.1) + 0.1  # scale to 0.1 - 1.0, [N, N]
  inter_mask_constrastive_loss = (loss_weight * inverse_f_m_d).mean()

  # return alpha * intra_mask_smoothing_loss + beta * inter_mask_constrastive_loss
  # --------------------------------------
  # print(intra_mask_smoothing_loss, inter_mask_constrastive_loss)

  # make sure the feat norm is one
  feat_norm_loss = (torch.norm(feat, p=2, dim=0) - 1.0).abs().mean()
  return alpha * intra_mask_smoothing_loss + beta * inter_mask_constrastive_loss + gamma * feat_norm_loss


def og_mask_constrastive_loss(feat,
                              mask,
                              alpha=1.0,
                              beta=1.0,
                              gamma=0.1,
                              tau=0.1,
                              mode='l2',
                              feat_norm_switch=False):
  """
  Input:
    feat: Feature map (d, h, w)
    mask: n bool Mask (n, h, w)
    alpha: weight for the intra-mask smoothing loss
    beta: weight for the inter-mask constrastive loss
  Output:
    loss
  """
  # tau = 0.2
  tau_inv = 1 / tau
  mask = mask.float()
  N = mask.shape[0]
  if feat_norm_switch:
    # feat = feat / torch.clamp_min(torch.norm(feat, p=2, dim=0, keepdim=True),
    #                               1e-6)
    feat = F.normalize(feat, p=2, dim=0)

  pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)

  # feat_mean: (n, d) <- sum(feat(1, d, h, w) * mask(n, 1, h, w))/(pixel_per_mask(n,1))
  feat_mean = torch.einsum('dhw,nhw->nd', feat, mask) / pixel_per_mask

  # --------------------------------------
  assert mode in ['l2', 'cos']
  if mode == 'l2':
    # feat-f_mean: (n, d, hw) <- feat(1, d, h, w) - feat_mean(n, d, 1)
    dist = rearrange(feat, 'd h w -> 1 d (h w)') - feat_mean.unsqueeze(
        2)  # (n, d, hw)
    mask_dist = (dist * rearrange(mask, 'n h w -> n 1 (h w)')).norm(
        p=2, dim=1)  # (n, hw)
  if mode == 'cos':
    # feat-f_mean: (n, d, hw) <- cos<feat(1, d, h*w), feat_mean(n, d, 1)>
    dist = 1.0 - rearrange(feat, 'd h w -> 1 d (h w)') * feat_mean.unsqueeze(
        2)  # (n, d, hw)
    mask_dist = (dist * rearrange(mask, 'n h w -> n 1 (h w)')).sum(
        dim=1)  # (n, hw)

  # intra_mask_smoothing_loss = mask_dist.mean()
  intra_mask_smoothing_loss = torch.exp(mask_dist * tau_inv).mean() - 1.0
  # intra_mask_smoothing_loss = -torch.log(
  #     torch.relu(tau - mask_dist) / tau + 1e-4).mean()
  # mask_dist = dist * rearrange(mask, 'n h w -> n 1 (h w)')  # (n, d, hw)
  # intra_mask_smoothing_loss = torch.norm(mask_dist, p=2, dim=1).mean()
  # --------------------------------------
  # inter_mask_constrastive_loss
  if mode == 'l2':
    # feat_mean_dist: (n, n, d) <- feat_mean(1, n, d) - feat_mean(n, 1, d)
    feat_mean_dist = torch.norm(feat_mean.unsqueeze(1) - feat_mean.unsqueeze(0),
                                p=2,
                                dim=2)  # (n, n)
  if mode == 'cos':
    # feat_mean_dist: (n, n, d) <- cos<feat_mean(1, n, d), feat_mean(n, 1, d)>
    feat_mean_dist = 1.0 - (feat_mean.unsqueeze(1) *
                            feat_mean.unsqueeze(0)).sum(dim=2)  # (n, n)
  eye = ~torch.eye(feat_mean_dist.shape[0], device=feat.device).bool()  # (n, n)
  feat_mean_dist = feat_mean_dist[eye].view(
      feat_mean_dist.shape[0], feat_mean_dist.shape[0] - 1)  # (n, n-1)

  # inverse_f_m_d = 1.0 / (feat_mean_dist + epsilon)
  # inverse_f_m_d = F.relu(-torch.log(feat_mean_dist / tau + 1e-3))
  # inverse_f_m_d = -torch.log(feat_mean_dist / tau + 1e-3)
  # inverse_f_m_d = torch.clamp_max(inverse_f_m_d, 10)
  inverse_f_m_d = torch.exp(-feat_mean_dist * tau_inv)

  # ReWeight it
  sorted_indices = inverse_f_m_d.argsort().argsort()
  loss_weight = (sorted_indices.float() /
                 (N - 1)) * (1.0 - 0.1) + 0.1  # scale to 0.1 - 1.0, [N, N]
  inter_mask_constrastive_loss = (loss_weight * inverse_f_m_d).mean()

  # return alpha * intra_mask_smoothing_loss + beta * inter_mask_constrastive_loss
  # --------------------------------------
  # print(intra_mask_smoothing_loss, inter_mask_constrastive_loss)

  # make sure the feat norm is one
  # feat_norm_loss = (torch.norm(feat, p=2, dim=0) - 1.0).abs().mean()
  return alpha * intra_mask_smoothing_loss + beta * inter_mask_constrastive_loss
  # + gamma * feat_norm_loss


# ------------------------------------------------------------------------------------
# Feature Distillation Loss


def distortion_mask_loss(f_dist,
                         d_dist,
                         mask,
                         alpha=1.0,
                         beta=0.001,
                         get_map=False):
  """
  Input:
    distortion: Feature map (d, h, w)
    mask: n bool Mask (n, h, w)
    alpha: weight for the intra-mask smoothing loss
  Output:
    loss
  """
  if f_dist.shape[1:] != mask.shape[1:]:
    f_dist = F.interpolate(f_dist.unsqueeze(0),
                           size=mask.shape[1:],
                           mode='bilinear',
                           align_corners=False).squeeze(0)
  # mask = mask.float()
  # N = mask.shape[0]
  # D = f_dist.shape[0]
  f_dist = f_dist.abs()
  # d_dist = 1 - (d_dist - d_dist.min()) / (d_dist.max() - d_dist.min() + 1e-8)
  # return alpha * (f_dist * (d_dist**0.5).clamp(0.0, 1.0)).mean()
  if mask is not None:
    mask = mask.float()
    N = mask.shape[0]
    pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)
    # H, W = mask.shape[1:]
    # pixel_num_mask = (pixel_per_mask > H * W * 0.25).squeeze(1)  # (n,1)
    # mask[pixel_num_mask, :, :] = 0.0
    if get_map:
      dist_mean = torch.einsum('dhw,nhw->dhw', f_dist,
                               mask / pixel_per_mask.unsqueeze(1))  # (n, h, w)
      return alpha * (dist_mean)
    else:
      dist_mean = torch.einsum('dhw,nhw->nd', f_dist,
                               mask) / pixel_per_mask  # (n, d)
      return alpha * (dist_mean).mean()
  else:
    return alpha * f_dist if get_map else alpha * (f_dist).mean()


# ------------------------------------------------------------------------------------
# Clip Feature Distallation


def feat_map_process(feat, mask):
  """
  Input:
    feat: Feature map (f, h, w)
    mask: n bool Mask (n, h, w)
  Output:
    feat_in_mask: (n, f)
  """
  pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)

  feat_mean = torch.einsum('fhw,nhw,nf', feat, mask) / pixel_per_mask


# def lang_distil_loss(feat_map, mask, clip):
#   """
#   Input:
#     feat_map: Feature map (f, h, w)
#     mask: n bool Mask (n, h, w)
#     clip: Clip feature (n, f)
#   Output:
#     loss
#   """
#   mask = mask.float()
#   pixel_per_mask = mask.sum(dim=(1, 2)).unsqueeze(1)  # (n,1)
#   feat_mean = torch.einsum('fhw,nhw->nf', feat_map,
#                            mask) / pixel_per_mask  # (n, f)
#   feat_mean = F.normalize(feat_mean, p=2, dim=1, eps=1e-6)
#   clip = F.normalize(clip, p=2, dim=1, eps=1e-6)
#   l_loss = l2_loss(feat_mean, clip)
#   cos_sim = 1 - torch.einsum('nf,nf->n', feat_mean, clip).mean()
#   # return (1. - cos_sim).mean()
#   return 50 * l_loss + 1.0 * cos_sim


def lang_distil_loss(feat_map, mask, clip):
  """
  Input:
    feat_map: Feature map (f, h, w)
    mask: n bool Mask (n, h, w)
    clip: Clip feature (n, f)
  Output:
    loss
  """
  # mask = mask.float()
  feat_map = F.normalize(feat_map, p=2, dim=0, eps=1e-6)
  clip = F.normalize(clip, p=2, dim=1, eps=1e-6)
  clip = torch.einsum('nf,nhw->fhw', clip, mask)
  # clip = (clip.unsqueeze(-1).unsqueeze(-1) * mask.unsqueeze(1)).sum(dim=0)
  feat_map = torch.einsum('fhw,nhw->fhw', feat_map, mask)
  # feat_map = feat_map * mask.sum(dim=0)

  # cos_sim = torch.einsum('fhw,fhw->n', feat_map, clip)
  # L1 loss
  l_loss = l1_loss(feat_map, clip)
  cos_loss = 1 - F.cosine_similarity(feat_map, clip, dim=0).mean()
  return 50 * l_loss + 1.0 * cos_loss
  # 0.010 is the min loss
  # return (1. - cos_sim).mean()
  # return cos_sim.mean()
