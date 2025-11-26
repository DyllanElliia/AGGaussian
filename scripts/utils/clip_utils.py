from tqdm import tqdm
import torch, torchvision
import torch.nn as nn
from typing import Tuple, Type
# from .clip_utils import OpenCLIPNetwork
import numpy as np
import clip
from dataclasses import dataclass, field


@dataclass
class OpenCLIPNetworkConfig:
  _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
  clip_model_type: str = 'ViT-B/32'
  # ViT-B-16-laion2b_s34b_b88k.bin, laion/CLIP-ViT-H-14-laion2B-s32B-b79K
  # clip_model_pretrained: str = "./clip_ckpt/ViT-H-14-laion2B-s32B-b79K.bin"
  # clip_model_pretrained: str = "./clip_ckpt/ViT-B-16-laion2b_s34b_b88k.bin"
  clip_n_dims: int = 512
  negatives: Tuple[str] = (
      "object",
      "things",
      "stuff",
      "texture",
  )
  # negatives: Tuple[str] = (
  #     'background',
  #     'brightness',
  #     'color',
  #     'context',
  #     'contrast',
  #     'diffuseness',
  #     'environment',
  #     'form',
  #     'geometry',
  #     'highlight',
  #     'hue',
  #     'illumination',
  #     'lighting',
  #     'material',
  #     'object',
  #     'occlusion',
  #     'opacity',
  #     'pattern',
  #     'reflection',
  #     'repetition',
  #     'rhythm',
  #     'saturation',
  #     'scene',
  #     'shade',
  #     'shadow',
  #     'shape',
  #     'specularity',
  #     'structure',
  #     'stuff',
  #     'surface',
  #     'symmetry',
  #     'texture',
  #     'things',
  #     'tone',
  #     'transparency',
  #     'value'
  # )
  positives: Tuple[str] = ("",)


class OpenCLIPNetwork(nn.Module):

  def __init__(self, config: OpenCLIPNetworkConfig, device="cuda"):
    super().__init__()
    self.config = config
    self.device = device
    self.process = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    # model, _, _ = open_clip.create_model_and_transforms(
    #     self.config.clip_model_type,  # e.g., ViT-B-16
    #     pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
    #     precision="fp16",
    # )
    self.model, clip_process = clip.load(self.config.clip_model_type,
                                         device=self.device)
    print("Self process", self.process)
    print("Clip process", clip_process)
    self.model.eval()
    self.tokenizer = clip.tokenize
    self.clip_n_dims = self.config.clip_n_dims

    self.positives = self.config.positives
    self.negatives = self.config.negatives
    with torch.no_grad():
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
      self.pos_embeds = self.model.encode_text(tok_phrases)
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.negatives]).to(self.device)
      self.neg_embeds = self.model.encode_text(tok_phrases)
    self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

    print('Embedding dimension', self.pos_embeds.shape[1])
    assert (
        self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
    ), "Positive and negative embeddings must have the same dimensionality"
    assert (
        self.pos_embeds.shape[1] == self.clip_n_dims
    ), f"Embedding dimensionality ({self.clip_n_dims}) must match the model dimensionality ({self.pos_embeds.shape[1]})"

  @property
  def name(self) -> str:
    return "openclip_{}".format(self.config.clip_model_type)

  @property
  def embedding_dim(self) -> int:
    return self.config.clip_n_dims

  def gui_cb(self, element):
    self.set_positives(element.value.split(";"))

  def set_positives(self, text_list):
    self.positives = text_list
    with torch.no_grad():
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
      self.pos_embeds = self.model.encode_text(tok_phrases)
    self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

  def set_positive_with_template(self, text, template):
    self.positives = [t.format(text) for t in template]
    with torch.no_grad():
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
      self.pos_embeds = self.model.encode_text(tok_phrases)
    self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    self.pos_embeds = self.pos_embeds.mean(dim=0, keepdim=True)
    self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
    # print(self.pos_embeds.shape)
    # self.pos_embeds = self.pos_embeds.mean(dim = 0)[None, :]
    # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

  def set_positives(self, text_list):
    self.positives = text_list
    with torch.no_grad():
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
      self.pos_embeds = self.model.encode_text(tok_phrases)
    self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

  def set_positives_with_template(self, text, template):
    self.positives = []
    for txt in text:
      for t in template:
        self.positives.append(t.format(txt))
    # self.positives = [t.format(text) for t in template]
    with torch.no_grad():
      tok_phrases = torch.cat(
          [self.tokenizer(phrase) for phrase in self.positives]).to(self.device)
      self.pos_embeds = self.model.encode_text(tok_phrases)
      self.pos_embeds = self.pos_embeds.view(len(text), len(template), -1)
      self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
      self.pos_embeds = self.pos_embeds.mean(dim=1)
      self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

  def get_relevancy(self, embed: torch.Tensor,
                    positive_id: int) -> torch.Tensor:
    phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
    p = phrases_embeds.to(embed.dtype)  # phrases x 512
    output = torch.mm(embed, p.T)  # rays x phrases
    positive_vals = output[..., positive_id:positive_id + 1]  # rays x 1
    negative_vals = output[..., len(self.pos_embeds):]  # rays x N_phrase
    repeated_pos = positive_vals.repeat(1,
                                        len(self.negatives))  # rays x N_phrase
    sims = torch.stack((repeated_pos, negative_vals),
                       dim=-1)  # rays x N-phrase x 2
    softmax = torch.softmax(10 * sims, dim=-1)  # N_image_features x N_negs x 2

    # strange implementation
    # best_id = softmax[..., 0].argmin(dim=1)  # N_image_features
    # print(best_id.shape,"best id shape")
    # return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    lowest_score = softmax[..., 0].min(dim=1)[0]
    return torch.stack([lowest_score, 1 - lowest_score], dim=-1)

  def get_relevancy_with_template(self, embed: torch.Tensor) -> torch.Tensor:
    phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
    p = phrases_embeds.to(embed.dtype)  # phrases x 512
    output = torch.mm(embed, p.T)  # rays x phrases
    positive_vals = output[..., :len(self.pos_embeds), None]  # rays x N_pos x 1
    negative_vals = output[..., None, len(self.pos_embeds):]  # rays x 1 x N_neg

    repeated_pos = positive_vals.repeat(1, 1, len(
        self.negatives))  # rays x N_pos x N_neg
    repeated_neg = negative_vals.repeat(1, len(self.pos_embeds),
                                        1)  # rays x N_pos x N_neg
    sims = torch.stack((repeated_pos, repeated_neg),
                       dim=-1)  # rays x N_pos x N_neg x 2
    softmax = torch.softmax(10 * sims,
                            dim=-1)  # N_image_features x N_pos x N_neg x 2

    lowest_score = softmax[..., 0].min(dim=-1)[0]  # N_image_features x N_pos
    # lowest_score = lowest_score.mean(dim = 1)
    return torch.stack([lowest_score, 1 - lowest_score], dim=-1)

  def get_segmentation(self, embed: torch.Tensor) -> torch.Tensor:
    # print(embed.shape, self.pos_embeds.shape)
    score = torch.einsum('nc,pc->np', embed, self.pos_embeds)
    return score

  def encode_image(self, input):
    processed_input = self.process(input).half()
    return self.model.encode_image(processed_input)

  def encode_text(self, text):
    t = self.tokenizer(text).to(self.device)
    return self.model.encode_text(t)


def load_clip(device="cuda"):
  return OpenCLIPNetwork(OpenCLIPNetworkConfig, device=device)


@torch.no_grad()
def get_features_from_image_and_masks(clip_model,
                                      image: np.array,
                                      masks: torch.tensor,
                                      background=1.):
  image_shape = image.shape[:2]
  if masks.shape[1] != image_shape[0] or masks.shape[2] != image_shape[1]:
    print('Resizing masks from {} to {}'.format(masks.shape, image_shape))
    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).float(),
                                            image_shape,
                                            mode='bilinear').squeeze(0)

    masks[masks > 0.5] = 1
    masks[masks != 1] = 0

  original_image = torch.from_numpy(image)[None].to(masks.device)

  masked_images = masks[:, :, :, None] * original_image + (
      1 - masks[:, :, :, None]) * 255. * background
  masks = masks.cpu()

  bboxes = torchvision.ops.masks_to_boxes(masks)

  bbox_heights = bboxes[:, 2] - bboxes[:, 0]
  bbox_widths = bboxes[:, 3] - bboxes[:, 1]

  bboxes = bboxes.int().tolist()

  cropped_seg_image_features1x = []

  for seg_idx in range(len(bboxes)):
    with torch.no_grad():
      tmp_image = masked_images[seg_idx][bboxes[seg_idx][1]:bboxes[seg_idx][3],
                                         bboxes[seg_idx][0]:bboxes[seg_idx]
                                         [2], :]  # (H, W, C)
      try:
        import matplotlib.pyplot as plt
        plt.imshow(tmp_image.cpu().numpy() / 255.0)
        plt.imsave("tmp.jpg", tmp_image.cpu().numpy() / 255.0)
      except:
        print(tmp_image)
      tmp_image = tmp_image.cuda()
      masked_image_clip_features = clip_model.encode_image(
          tmp_image[None, ...].permute([0, 3, 1, 2]) / 255.0)
      cropped_seg_image_features1x.append(masked_image_clip_features.cpu())

      # 1 H W C
      # tmp_image = masked_images[seg_idx][bboxes2x[seg_idx][1]:bboxes2x[seg_idx][3], bboxes2x[seg_idx][0]:bboxes2x[seg_idx][2], :]
      # tmp_image = tmp_image.cuda()
      # masked_image_clip_features2x = clip_model.encode_image(tmp_image[None,...].permute([0,3,1,2]) / 255.0)
      # cropped_seg_image_features2x.append(masked_image_clip_features2x)

      # tmp_image = masked_images[seg_idx][bboxes4x[seg_idx][1]:bboxes4x[seg_idx][3], bboxes4x[seg_idx][0]:bboxes4x[seg_idx][2], :]
      # masked_image_clip_features4x = clip_model.encode_image(tmp_image[None,...].permute([0,3,1,2]) / 255.0)
      # cropped_seg_image_features4x.append(masked_image_clip_features4x)

  cropped_seg_image_features1x = torch.cat(cropped_seg_image_features1x, dim=0)
  # cropped_seg_image_features2x = torch.cat(cropped_seg_image_features2x, dim=0)
  # cropped_seg_image_features4x = torch.cat(cropped_seg_image_features4x, dim=0)

  return cropped_seg_image_features1x
