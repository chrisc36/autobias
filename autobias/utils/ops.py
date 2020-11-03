import logging

import numpy as np
import scipy
import torch
from scipy.special import expit
from torch.autograd import Variable
from torch.nn import functional as F

from autobias.utils import py_utils


def elementwise_logsumexp(x, y):
  return torch.max(x, y) + torch.log1p(torch.exp(-torch.abs(x - y)))


def elementwise_logsumexp_np(x, y):
  return np.maximum(x, y) + np.log1p(np.exp(-np.abs(x - y)))


def sigmoid_np(x):
  return expit(x)


def softmax_np(x, axis):
  return scipy.special.softmax(x, axis)


def log_sigmoid(x):
  return -F.softplus(-x)


def weighted_mean(x, weights=None):
  if weights is None:
    return x.mean()
  return (x * weights).sum() / weights.sum()


def get_fn(fn_name):
  if fn_name == "relu":
    return F.relu
  if fn_name == "tanh":
    return torch.tanh
  elif fn_name is None:
    return lambda x: x
  else:
    raise NotImplementedError(fn_name)


def numpy(x):
  if x is None:
    return x
  if isinstance(x, Variable):
    x = x.detach()
  return x.cpu().numpy()


def max_pool(x, x_mask):
  if x_mask is not None:
    x *= x_mask.binary_mask.unsqueeze(-1).to(x.dtype)
  return x.max(-2)[0]


def mask(x, mask):
  if mask is None:
    return x
  if isinstance(mask, Mask):
    mask = mask.binary_mask
  if len(mask.size()) < len(x.size()):
    mask = torch.unsqueeze(mask, -1)
  mask = mask.to(x.dtype)
  return x * mask


def mask_logits(x, mask):
  if mask is None:
    return x
  if isinstance(mask, Mask):
    mask = mask.binary_mask
  if len(mask.size()) < len(x.size()):
    mask = torch.unsqueeze(mask, -1)
  mask = mask.to(x.dtype)

  # use smallish value to ensure this does not become NaN in fp16 mode
  return x * mask - (1 - mask) * 10000.0


def load_and_log(model, state_dict):
  missing_keys = []
  unexpected_keys = []
  error_msgs = []
  # copy state_dict so _load_from_state_dict can modify it
  metadata = getattr(state_dict, '_metadata', None)
  state_dict = state_dict.copy()
  if metadata is not None:
    state_dict._metadata = metadata

  def load(module, prefix=''):
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    module._load_from_state_dict(
      state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    for name, child in module._modules.items():
      if child is not None:
        load(child, prefix + name + '.')
  load(model)
  name = model.__class__.__name__

  if len(missing_keys) > 0:
    to_show = missing_keys
    if len(to_show) > 2:
      to_show = missing_keys[:1]+missing_keys[-1:]
    logging.info(f"{len(missing_keys)} weights of {name} not initialized "
    f"from pretrained model, including: {to_show}")
  if len(unexpected_keys) > 0:
    to_show = unexpected_keys
    if len(to_show) > 2:
      to_show = unexpected_keys[:1]+unexpected_keys[-1:]
    logging.info(f"{len(unexpected_keys)}/{len(state_dict)} weights of {name} pre-trained "
    f"model not found, including: {to_show}")
  if len(error_msgs) > 0:
    raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
      name, "\n\t".join(error_msgs)))


def cross_entropy(logits, targets, weights=None):
  if weights is None:
    return F.cross_entropy(logits, targets)
  else:
    loss = F.cross_entropy(logits, targets, reduction='none')
    return weighted_mean(loss, weights)


def one_hot(label, n_classes):
  batch = len(label)
  one_hot = torch.zeros(batch, n_classes, device=label.device, dtype=torch.float32)
  one_hot.scatter_add_(
    1, label.unsqueeze(1),torch.ones(batch, 1, device=label.device, dtype=torch.float32))
  return one_hot


def smooth_one_hot(label, n_classes, smooth):
  batch = len(label)
  correct_prob = 1.0 - smooth
  other_prob = smooth / (n_classes - 1)
  soft_labels = torch.full(
    (batch, n_classes), fill_value=other_prob, device=label.device, dtype=torch.float32)
  soft_labels.scatter_add_(
    1, label.unsqueeze(1),
    torch.full((batch, 1), (correct_prob - other_prob), device=label.device,
               dtype=torch.float32))
  return soft_labels


def multilabel_binary_cross_entropy(logits, targets, weights=None):
  if weights is None:
    return F.binary_cross_entropy_with_logits(logits, targets) * targets.size(1)
  else:
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    return (loss * weights.unsqueeze(1)).sum() / weights.sum()


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def scatter_add_dim0(x, out_dim, ix):
  last_dims = x.size()[1:]
  out = torch.zeros(*((out_dim,) + last_dims), device=x.device, dtype=x.dtype)
  sz = (ix.size(0), ) + tuple(1 for _ in range(len(last_dims)))
  ix = ix.view(sz).expand(*((ix.size(0),) + last_dims))
  return out.scatter_add_(0, ix, x)


def multi_prediction_loss(logits, labels, per_example=False):
  norms = torch.logsumexp(logits, -1)  # [batch, n_patches]
  label_scores = logits[
                 torch.arange(0, logits.size(0), device=logits.device), :, labels
                 ]
  if per_example:
    return (norms - label_scores).mean(1)
  else:
    return (norms - label_scores).mean()


class Mask:
  """Used to store masks for ragged sequences"""
  def __init__(self, binary_mask, sequence_lens, max_len=None):
    self._max_len = max_len
    self._binary_mask = binary_mask
    self._sequence_lens = sequence_lens

  @property
  def sequence_lens(self):
    if self._sequence_lens is None:
      self._sequence_lens = self._binary_mask.long().sum(1)
    return self._sequence_lens

  @property
  def binary_mask(self):
    if self._binary_mask is None:
      seq_lens = self.sequence_lens
      ix = torch.arange(self._max_len, device=seq_lens.device, dtype=seq_lens.dtype)
      self._binary_mask = (ix.expand(seq_lens.size(0), self._max_len) < seq_lens.unsqueeze(1)).float()
    return self._binary_mask

  def pin_memory(self):
    return Mask(
      None if self._binary_mask is None else self._binary_mask.pin_memory(),
      self.sequence_lens.pin_memory(), self._max_len)

  def to(self, device):
    return Mask(
      None if self._binary_mask is None else self._binary_mask.to(device),
      self.sequence_lens.to(device), self._max_len)


def build_masks(lens):
  max_l = max(lens)
  out = np.zeros((len(lens), max_l), dtype=np.float32)
  for i, l in enumerate(lens):
    out[i, :l] = 1
  return Mask(torch.as_tensor(out), torch.LongTensor(lens))


def collate_list(lst):
  if isinstance(lst[0], (int, np.int32, np.int64)):
    return torch.LongTensor(lst)

  max_l = max(x.shape[0] for x in lst)
  shape = (len(lst), max_l) + lst[0].shape[1:]
  dtype = np.int64 if lst[0].dtype == np.int32 else lst[0].dtype
  out = np.zeros(shape, dtype=dtype)
  for i, x in enumerate(lst):
    out[i, :len(x)] = x
  return torch.as_tensor(out)


def collate_list_of_tuples(arr):
  return [collate_list(x) for x in py_utils.transpose_lists(arr)]


def collate_flat_dict(examples):
  return {k: collate_list([ex[k] for ex in examples]) for k in examples[0]}


def soft_cross_entropy(logprobs, target_logprobs):
  return -(logprobs * torch.exp(target_logprobs)).sum(1).mean(0)
