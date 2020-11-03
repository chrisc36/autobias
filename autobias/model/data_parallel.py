import operator
import warnings
from itertools import chain

import torch
from torch import nn
from torch.nn.parallel import replicate, parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather

from autobias.model.model import TrainOutput
from autobias.utils import py_utils


def _get_device_index(device, optional=False):
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a CUDA device. Note that for a CUDA device without a specified index,
    i.e., ``torch.device('cuda')``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default CUDA
    device if :attr:`optional` is ``True``.
    """
    if isinstance(device, torch._six.string_classes):
        device = torch.device(device)
    if isinstance(device, torch.device):
        dev_type = device.type
        if device.type != 'cuda':
            raise ValueError('Expected a cuda device, but got: {}'.format(device))
        device_idx = device.index
    else:
        device_idx = device
    if device_idx is None:
        if optional:
            # default cuda device index
            return torch.cuda.current_device()
        else:
            raise ValueError('Expected a cuda device with a specified index '
                             'or an integer, but got: '.format(device))
    return device_idx


def _check_balance(device_ids):
  imbalance_warn = """
  There is an imbalance between your GPUs. You may want to exclude GPU {} which
  has less than 75% of the memory or cores of GPU {}. You can do so by setting
  the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
  environment variable."""
  device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
  dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

  def warn_imbalance(get_prop):
    values = [get_prop(props) for props in dev_props]
    min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
    max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
    if min_val / max_val < 0.75:
      warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
      return True
    return False

  if warn_imbalance(lambda props: props.total_memory):
    return
  if warn_imbalance(lambda props: props.multi_processor_count):
    return


class DataParallel(nn.Module):
  """
  Our version of `torch.nn.DataParallel`, modified to support models that have a 'batch_loss`,
  meaning a loss that must be computed across the entire batch, not per example.

  For those kinds of models, we build features on each GPU, aggregate those feature on a single
  GPU, and then use them to compute the loss on that GPU. We use this to support the argmin
  operations that need to computed across the entire batch.

  This version also return an output per a gpu, instead of gathering the outputs on to one
  GPU as the torch version does.
  """

  def __init__(self, model, presplit_input, device_ids=None,
               output_device=None):
    super().__init__()
    self.presplit_input = presplit_input
    if not torch.cuda.is_available():
      raise ValueError()

    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]

    self.model = model
    self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    self.output_device = _get_device_index(output_device, True)
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

    _check_balance(self.device_ids)

    if len(self.device_ids) == 1:
      raise ValueError()
    self._replicas = None

  def forward(self, *inputs, **kwargs):

    for t in chain(self.model.parameters(), self.model.buffers()):
      if t.device != self.src_device_obj:
        raise RuntimeError("module must have its parameters and buffers "
                           "on device {} (device_ids[0]) but found one of "
                           "them on device: {}".format(self.src_device_obj, t.device))

    if not self.presplit_input:
      inputs = self.scatter(inputs, None, self.device_ids)[0]
    # else we trust the client will scatter the input for us

    if self.training:
      replicas = self.replicate(self.device_ids[:len(inputs)])
    else:
      replicas = self._replicas[:len(inputs)]

    if not self.training:
      # No need to gather since we return one output for a GPU
      return self.parallel_apply(replicas, inputs)

    if self.model.has_batch_loss():
      # Build the features on each GPU
      features = self.parallel_apply(replicas, inputs, dict(mode="features"))

      if kwargs.get("mode") == "features":
        # Client just wants the features
        return gather(features, self.output_device)

      # Aggregate
      features = gather(features, self.output_device)

      # Have first replica compute the batch loss using the aggregated features
      labels = gather([x[-1] for x in inputs], self.output_device)
      out = replicas[0](features, labels, mode="loss")
      return out
    else:
      outputs = self.parallel_apply(replicas, inputs)
      if outputs[0].monitor is not None:
        monitor = {}
        for k in outputs[0].monitor.keys():
          monitor[k] = gather([x.monitor[k].unsqueeze(0) for x in outputs], self.output_device).mean()
      else:
        monitor = None
      return TrainOutput(gather([x.loss.unsqueeze(0) for x in outputs], self.output_device).mean(),
                         monitor)

  def train(self, mode=True):
    super().train(mode)
    if mode:
      self._replicas = None
    elif self._replicas is None:
      self._replicas = self.replicate(self.device_ids)

  def replicate(self, device_ids):
    return replicate(self.model, device_ids)

  def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=0)

  def parallel_apply(self, replicas, inputs, kargs=None):
    if kargs is not None:
      kargs = tuple(kargs for _ in inputs)
    return parallel_apply(replicas, inputs, kargs, self.device_ids[:len(replicas)])

