import torch
import torch.nn as nn

from .. import GaussianModel


class OptimizerModifier:

  def __init__(self, logger, model: GaussianModel):
    self.logger = logger
    self.model = model

  # add the tensors to the optimizer via the cat operation
  def cat(self, optimizer, tensors_dict):
    self.logger.info("Cat tensors in optimizer.")
    optimizable_tensors = {}
    for group in optimizer.param_groups:
      assert len(group["params"]) == 1
      extension_tensor = tensors_dict[group["name"]]
      stored_state = optimizer.state.get(group['params'][0], None)
      if stored_state is not None:
        stored_state["exp_avg"] = torch.cat(
            (stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
            dim=0)
        stored_state["exp_avg_sq"] = torch.cat(
            (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
            dim=0)

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(
            torch.cat((group["params"][0], extension_tensor),
                      dim=0).requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(
            torch.cat((group["params"][0], extension_tensor),
                      dim=0).requires_grad_(True))
        optimizable_tensors[group["name"]] = group["params"][0]
      # self.logger.info(
      #     f"Cat {group['name']} in optimizer, shape: {group['params'][0].shape}"
      # )

    return optimizable_tensors

  # replace the name's tensors in the optimizer with the new tensors.
  # If mask is not None, only replace the variables where its mask is True.
  def replace_one(self, optimizer, tensor, name, mask=None):
    self.logger.info(f"Replace {name} in optimizer.")
    optimizable_tensors = {}
    for group in optimizer.param_groups:
      if group["name"] == name:
        stored_state = optimizer.state.get(group['params'][0], None)
        group_params = group["params"][0]

        if mask is None:
          stored_state["exp_avg"] = torch.zeros_like(tensor)
          stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
          group_params = tensor
        else:
          stored_state["exp_avg"][mask] = torch.zeros_like(tensor)
          stored_state["exp_avg_sq"][mask] = torch.zeros_like(tensor)
          group_params[mask] = tensor

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(group_params.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors

  def replace(self, optimizer, tensors_dict):
    self.logger.info("Replace tensors in optimizer.")
    optimizable_tensors = {}
    for group in optimizer.param_groups:
      assert len(group["params"]) == 1
      tensor = tensors_dict[group["name"]]
      stored_state = optimizer.state.get(group['params'][0], None)
      if stored_state is not None:
        stored_state["exp_avg"] = torch.zeros_like(tensor)
        stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state

        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
        optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors

  # remove the variables from the optimizer if its mask is False
  def remove(self, optimizer, mask):
    self.logger.info("Remove tensors in optimizer.")
    optimizable_tensors = {}
    for group in optimizer.param_groups:
      stored_state = optimizer.state.get(group['params'][0], None)
      # self.logger.info(f"Remove {group['name']} in optimizer.")
      if stored_state is not None:
        stored_state["exp_avg"] = stored_state["exp_avg"][mask]
        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(
            (group["params"][0][mask].requires_grad_(True)))

        optimizer.state[group['params'][0]] = stored_state
        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(
            group["params"][0][mask].requires_grad_(True))
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors

  # reorder the variables in the optimizer according to the indices (N,)
  def reorder(self, optimizer, indices):
    self.logger.info("Reorder tensors in optimizer.")
    optimizable_tensors = {}
    for group in optimizer.param_groups:
      stored_state = optimizer.state.get(group['params'][0], None)
      if stored_state is not None:
        stored_state["exp_avg"] = stored_state["exp_avg"][indices]
        stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][indices]

        del optimizer.state[group['params'][0]]
        group["params"][0] = nn.Parameter(
            (group["params"][0][indices].requires_grad_(True)))

        optimizer.state[group['params'][0]] = stored_state
        optimizable_tensors[group["name"]] = group["params"][0]
      else:
        group["params"][0] = nn.Parameter(
            group["params"][0][indices].requires_grad_(True))
        optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors
