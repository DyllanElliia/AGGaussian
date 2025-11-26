import torch
import torch.nn as nn
import torch.nn.functional as F
"""
We did not use them in this work.
"""


class MLP(nn.Module):

  def __init__(self,
               input_dim,
               output_dim,
               hidden_units,
               hidden_activation='relu',
               output_activation=None,
               init_weight_type=None):
    super(MLP, self).__init__()

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_units = hidden_units
    self.hidden_activation = getattr(F, hidden_activation)
    if output_activation is not None:
      self.output_activation = getattr(torch, output_activation)
    else:
      self.output_activation = lambda x: x

    self.layers = nn.ModuleList()
    prev_h = input_dim
    for h in hidden_units:
      self.layers.append(nn.Linear(prev_h, h))
      prev_h = h
    self.layers.append(nn.Linear(prev_h, output_dim))

    if init_weight_type is not None:
      self.init_weight(init_weight_type)

  def init_weight(self, init_weight_type):
    if init_weight_type == 'xavier':
      print('xavier')
      for layer in self.layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    elif init_weight_type == 'kaiming':
      print('kaiming')
      for layer in self.layers:
        nn.init.kaiming_uniform_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    elif init_weight_type == 'orthogonal':
      print('orthogonal')
      for layer in self.layers:
        nn.init.orthogonal_(layer.weight)
        nn.init.constant_(layer.bias, 0)
    elif init_weight_type == 'normal':
      print('normal')
      for layer in self.layers:
        nn.init.normal_(layer.weight, 0, 0.02)
        nn.init.constant_(layer.bias, 0)
    elif init_weight_type == 'uniform':
      print('uniform')
      for layer in self.layers:
        nn.init.uniform_(layer.weight, -0.02, 0.02)
        nn.init.constant_(layer.bias, 0)
    else:
      raise ValueError('Invalid init_weight_type')

  def forward(self, x):
    for layer in self.layers[:-1]:
      x = self.hidden_activation(layer(x))
    return self.output_activation(self.layers[-1](x))


def get_mlp(dim=[2, 64, 64, 1],
            hidden_activation='relu',
            output_activation=None,
            mlp_init_method=None):
  return MLP(dim[0], dim[-1], dim[1:-1], hidden_activation, output_activation,
             mlp_init_method)
