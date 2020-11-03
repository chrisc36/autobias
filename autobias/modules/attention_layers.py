import torch
from torch import nn
from torch.nn import functional as F

from autobias.modules.layers import Layer
from autobias.utils import ops


class DotAttention(Layer):
  def __init__(self, dim, transposed_priors=True):
    super().__init__()
    self.dim = dim
    self.transposed_priors = transposed_priors
    self.w1 = nn.Parameter(torch.Tensor(dim))
    self.w2 = nn.Parameter(torch.Tensor(dim))
    if transposed_priors:
      self.w3 = nn.Parameter(torch.Tensor(dim))
    else:
      self.w3 = None

  def reset_parameters(self):
    self.w1.data.fill_(1/self.dim)
    self.w2.data.zero_()
    if self.w2 is not None:
      self.w2.data.zero_()

  def get_scores(self, p, q):
    # [batch, p_len, 1]

    p_weights = p * self.w2.unsqueeze(0).unsqueeze(0)
    # [batch, p_len, dim] * [batch, dim, q_len] -> [batch, p_len, q_len]
    matrix = torch.matmul(p_weights, q.transpose(1, 2))

    q_priors = torch.matmul(q, self.w2.unsqueeze(1))
    q_priors = q_priors.squeeze(2).unsqueeze(1)
    matrix += q_priors

    if self.transposed_priors:
      p_priors = torch.matmul(p, self.w2.unsqueeze(1))
      matrix += p_priors
    return matrix

  def forward(self, tensor_1, tensor_2, mask1, mask2):
    atten = self.get_scores(tensor_1, tensor_2)
    mask = mask1.binary_mask.unsqueeze(2) * mask2.binary_mask.unsqueeze(1)
    return ops.mask_logits(atten, mask)


class AttentionBiFuse(Layer):
  def __init__(self, attention, include_concat=True):
    super().__init__()
    self.attention = attention
    self.include_concat = include_concat

  def reset_parameters(self):
    pass

  def forward(self, q, p, q_mask, p_mask):
    # [batch, p_len, q_len]
    atten = self.attention(q, p, q_mask, p_mask)

    # [batch, p_len, dim]
    q_to_p = torch.matmul(F.softmax(atten, -1), p)
    p_top_q = torch.matmul(F.softmax(atten.transpose(1, 2), -1), q)

    return torch.cat([q, q_to_p, q_to_p*q], -1), torch.cat([p, p_top_q, p_top_q * p], -1)

