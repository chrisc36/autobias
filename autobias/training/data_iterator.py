from typing import Union

from torch.utils.data import DataLoader

from autobias.training.data_batcher import SamplerSpec, SubsetSampler
from autobias.utils import py_utils
from autobias.utils.configured import Configured


class DataIterator(Configured):
  """
  Builds iterators over data, this abstraction is used so we can build tricky iterators like
  the ones used for VQA
  """

  def build_iterator(self, data, collate_fn, split_batch=None):
    raise NotImplementedError()

  def build_iterator_for_model(self, data, model, split_batch=None):
    return self.build_iterator(data, model.get_collate_train_fn(), split_batch)


class TorchDataIterator(DataIterator):
  def __init__(self, batching: Union[int, SamplerSpec], num_workers=0, pin_memory=False):
    self.num_workers = num_workers
    self.pin_memory = pin_memory
    self.batching = batching

  def build_iterator(self, data, model_collate_fn, split_batch=None):
    if not split_batch:
      collate_fn = model_collate_fn
    else:
      def collate_fn(x):
        return tuple(model_collate_fn(x) for x in py_utils.split(x, split_batch))

    if isinstance(self.batching, int):
      return DataLoader(
        data,
        shuffle=True,
        batch_size=self.batching,
        collate_fn=collate_fn,
        pin_memory=self.pin_memory,
        num_workers=self.num_workers
      )
    else:
      batcher = self.batching.build_sampler(data)
      if batcher is None:
        raise ValueError()
      # noinspection PyArgumentList
      return DataLoader(
          data,
          sampler=None,
          batch_size=1,
          batch_sampler=batcher,
          collate_fn=collate_fn,
          pin_memory=self.pin_memory,
          num_workers=self.num_workers
      )


class SampleIterator(TorchDataIterator):
  def __init__(self, n_samples, batch_size, num_workers=0, pin_memory=False, sort=False):
    self.n_samples = n_samples
    self.sort = sort
    self.batch_size = batch_size
    super().__init__(SubsetSampler(n_samples, batch_size, sort=sort), num_workers, pin_memory)
