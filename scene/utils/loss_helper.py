import os
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement

from einops import rearrange, repeat

from .. import GaussianModel


class LossHelper:

  def __init__(self, logger, model: GaussianModel):
    self.logger = logger
    self.model = model
    self.lossFunc_dict = {}

  def append(self, name, weight=1.0, lossFunc=lambda: 0.0):
    self.logger.info(f"Append loss function {name} with weight {weight}.")
    self.lossFunc_dict[name] = {"weight": weight, "lossFunc": lossFunc}

  def get_loss(self, name=None):
    if name is None:
      loss_dist = {}
      loss_sum = 0.0
      for name, lf in self.lossFunc_dict.items():
        loss = lf["weight"] * lf["lossFunc"]()
        loss_dist[name] = loss
        loss_sum += loss
      return loss_sum, loss_dist
    else:
      return self.lossFunc_dict[name]["weight"] * self.lossFunc_dict[name][
          "lossFunc"]()
