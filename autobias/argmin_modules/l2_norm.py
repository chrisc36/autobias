import numpy as np
import torch

from autobias.utils.configured import Configured


class L2NormPenalty(Configured):
  """
  L2 Norm penalty with second order gradient methods
  """

  def __init__(self, l2, epsilon=0):
    self.l2 = l2
    self.epsilon = epsilon

  def compute_loss(self, weights: torch.Tensor):
    weights = weights.view(-1)
    w_norm_squared = torch.dot(weights, weights) + self.epsilon
    if w_norm_squared == 0:
      return 0
    w_norm = torch.sqrt(w_norm_squared)
    return w_norm*self.l2

  def np_compute_loss(self, weights: np.ndarray):
    weights = weights.reshape(-1)
    w_norm_squared = np.dot(weights, weights) + self.epsilon
    if w_norm_squared == 0:
      return 0
    w_norm = np.sqrt(w_norm_squared)
    return w_norm*self.l2

  def np_compute_loss_and_grad(self, weights: np.ndarray):
    weights = weights.reshape(-1)
    w_norm_squared = np.dot(weights, weights) + self.epsilon
    if w_norm_squared == 0:
      return 0, np.zeros_like(weights)
    else:
      w_norm = np.sqrt(w_norm_squared)
      return w_norm*self.l2, self.l2 * weights / w_norm

  def compute_hessian(self, weights: torch.Tensor):
    weights = weights.view(-1)
    w_norm_squared = torch.dot(weights, weights) + self.epsilon
    if w_norm_squared == 0:
      return torch.zeros(len(weights), len(weights), dtype=weights.dtype)
    norm_dn = -torch.pow(w_norm_squared, -1.5)
    hessian = norm_dn * weights.unsqueeze(0) * weights.unsqueeze(1)
    hessian[torch.arange(len(weights)), torch.arange(len(weights))] += torch.rsqrt(w_norm_squared)
    return hessian * self.l2
