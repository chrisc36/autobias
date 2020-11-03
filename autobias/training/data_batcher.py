import logging
from collections import defaultdict
from typing import List, Iterable

import numpy as np
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data import RandomSampler as TorchRandomSampler

from autobias.utils import py_utils
from autobias.utils.configured import Configured


class SamplerSpec(Configured):
  """`Configured` object to control batching"""

  def build_sampler(self, data_source, model=None) -> Sampler:
    """
    :return: a batch-level `Sampler`
    """
    raise NotImplementedError()


def compute_batch_lens(rng, n, batch_size):
  """Build an array of batch sizes, such that the total of the sizes is `n`"""

  n_batches = (n + batch_size - 1) // batch_size
  batch_lens = np.full(n_batches, batch_size, np.int32)

  # How many extra examples are allocated in batch_lens
  extra = n_batches * batch_size - n

  # In rare cases we might have to reduce the size of all the batches
  # (ex. 21 examples with batches size of 10 -> batches of size 7)
  # FIXME should be able to avoid using a loop
  while extra > len(batch_lens):
    batch_lens -= 1
    extra -= len(batch_lens)

  # Reduce the len random set of batches by 1
  batch_lens[rng.choice(len(batch_lens), extra, False)] -= 1
  return batch_lens


def compute_batch_bounds(rng, n, batch_size):
  """Build an array of start/end indices to divide a list of size `n` into batches"""
  batch_lens = compute_batch_lens(rng, n, batch_size)

  batch_ends = np.cumsum(batch_lens)

  if batch_ends[-1] != n:
    raise RuntimeError()

  batch_starts = np.pad(batch_ends[:-1], [1, 0], "constant")

  bounds = np.stack([batch_starts, batch_ends], 1)
  return bounds


def balanced_merge_multi(lsts: Iterable[List]) -> List:
  """Merge lists while trying to keep them represented proportional to their lengths
  in any continuous subset of the output list
  """

  lens = np.array([len(x) for x in lsts], dtype=np.float64)
  target_ratios = lens / lens.sum()
  current_counts = np.zeros(len(lsts), dtype=np.int32)
  out = []
  lsts = [list(x) for x in lsts]
  while True:
    if len(out) == 0:
      next_i = np.argmax(target_ratios)
    else:
      # keep a track of the current mixing ratio, and add in the most under-repersnted list
      # each step
      next_i = np.argmin(current_counts / len(out) - target_ratios)
    current_counts[next_i] += 1
    lst = lsts[next_i]
    out.append(lst.pop())
    if len(lst) == 0:
      target_ratios = np.delete(target_ratios, next_i)
      current_counts = np.delete(current_counts, next_i)
      lsts = lsts[:next_i] + lsts[next_i+1:]
      if len(lsts) == 0:
        break

  return out[::-1]


class WrappedSampler(Sampler):
  def __init__(self, n, iter_fn):
    self.n = n
    self.iter_fn = iter_fn

  def __len__(self):
    return self.n

  def __iter__(self):
    return self.iter_fn()


class ShuffleSamplerSpec(SamplerSpec):
  """Randomly shuffle the input"""

  def __init__(self, batch_size, drop_last=False):
    self.batch_size = batch_size
    self.drop_last = drop_last

  def build_sampler(self, data_source, model=None):
    if len(data_source) == 0:
      raise ValueError()
    return BatchSampler(TorchRandomSampler(data_source), self.batch_size, self.drop_last)


class SubsetSampler(SamplerSpec):
  """Random subset of the data, used for partial evaluations during training"""

  def __init__(self, num_samples, batch_size, sort=False):
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.sort = sort

  def build_sampler(self, data_source, model=None):
    total = len(data_source) if self.num_samples is None else self.num_samples
    n = (total + self.batch_size - 1) // self.batch_size

    if self.sort:
      source_ix = np.argsort([x.get_len() for x in data_source])
    else:
      source_ix = np.arange(0, total)

    def iter():
      if self.num_samples is not None:
        ixs = np.random.choice(len(data_source), self.num_samples, replace=False)
        if self.sort:
          ixs.sort()
          ixs = source_ix[ixs]
      else:
        ixs = source_ix

      bounds = compute_batch_bounds(np.random, len(ixs), self.batch_size)
      for s, e in bounds:
        yield ixs[s:e]

    return WrappedSampler(n, iter)


class SortedBatchSampler(SamplerSpec):
  """Yields batch with elements of similar `element.get_len()` values"""

  def __init__(self,  batch_size, shuffle_batches=True):
    self.batch_size = batch_size
    self.shuffle_batches = shuffle_batches

  def build_sampler(self, data_source, model=None):
    n = len(data_source)
    n_batches = (n + self.batch_size - 1) // self.batch_size
    ixs = np.argsort([x.get_len() for x in data_source])

    def iter():
      bounds = compute_batch_bounds(np.random, n, self.batch_size)
      if self.shuffle_batches:
        np.random.shuffle(bounds)
      for s, e in bounds:
        yield ixs[np.arange(s, e)]

    return WrappedSampler(n_batches, iter)


class StratifiedSampler(SamplerSpec):
  """Yields batches with similar label distributions"""

  def __init__(self, batch_size, n_repeat=1):
    self.batch_size = batch_size
    self.n_repeat = n_repeat

  def build_sampler(self, data_source, model=None):
    if len(data_source) == 0:
      raise ValueError()
    label_to_ixs = defaultdict(list)
    n_batches = (len(data_source) + self.batch_size - 1) // self.batch_size
    for i, ex in enumerate(data_source):
      label_to_ixs[ex.label].append(i)
    ixs = list(label_to_ixs.values())
    bounds = compute_batch_bounds(np.random, len(data_source), self.batch_size)

    def iter():
      for _ in range(self.n_repeat):
        for ix in ixs:
          np.random.shuffle(ix)
        merged = np.array(balanced_merge_multi(ixs))
        for s, e in bounds:
          yield merged[s:e]

    return WrappedSampler(n_batches*self.n_repeat, iter)
