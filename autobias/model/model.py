from typing import List, Union, Any

from torch import nn

from autobias.datasets.dataset import Dataset
from autobias.modules.layers import Layer
from autobias.utils.configured import Configured


class TrainOutput:

  def __init__(self, loss, monitor=None):
    """
    :param loss: Loss to backprop on
    :param monitor: Dictionary of scalars we should log to tensorboard
    """
    self.loss = loss
    self.monitor = monitor


def _apply_if(x: nn.Module, fn):
  if fn(x):
    return
  for child in x.children():
    _apply_if(child, fn)


def init_weights(module):
  if isinstance(module, Layer):
    return module.reset_parameters()
  return False


class Model(nn.Module, Configured):
  """Our definition of a model, built on top of nn.Module

  Our models have additional APIs for pre-processing, collating, and saving state.

  Our models always take input in the form of (features, labels, **kwargs), and have a default
  reset_parameters routine that calls `reset_parameters` on each `Layer` inside the module.

  Typically models take care of turning the data into tensors, which are then passed to a
  `Predictor` sub-layer that does the actual computation
  """

  def preprocess_train(self, train: Dataset, other: List[Dataset], n_processes=None) -> List[List]:
    """Preprocess the training data, and any additional dataset we might evaluate on.

    :param train: Training data
    :param other: Any other dataset the model might be evaluated on
    :param n_processes: number of processes for multi-process pre-processing
    :return: For each dataset, a list of pre-processed examples
    """
    return self.preprocess_datasets([train] + other, n_processes)

  def preprocess_datasets(self, datasets: List[Dataset], n_processes=None) -> List[List]:
    """Preprocessing the input datasets, should only be called after the model is trained.

    :param datasets: Dataset the model will be evaluated on
    :param n_processes: number of processes for multi-process pre-processing
    :return: For each dataset, a list of pre-processed examples
    """
    raise NotImplementedError()

  def preprocess_dataset(self, datasets: Dataset, n_processes=None) -> List:
    return self.preprocess_datasets([datasets], n_processes)[0]

  def get_collate_fn(self):
    """Collate function to use on the evaluation data"""
    raise NotImplementedError()

  def get_collate_train_fn(self):
    """Collate function to use on the training data"""
    return self.get_collate_fn()

  def reset_parameters(self):
    for child in self.children():
      _apply_if(child, init_weights)

  def get_state(self):
    # By default just use state_dict, but allow sub-classes to tack in additional
    # state if needed (ex. what answer vocabulary we decided to use)
    return self.state_dict()

  def load_state(self, state):
    self.load_state_dict(state)

  def has_batch_loss(self):
    """Does the loss need to be computed across the entire batch (meaning it shouldn't be
    computed independently on different GPUs and then aggregated).

    Our `autobias.model.data_parallel.DataParallel` knows how to special-case these kinds of
    losses when training on them in a multi-gpu setting.
    """
    return False

  def forward(self, features, label=None, **kwargs) -> Union[TrainOutput, Any]:
    raise NotImplementedError()


class Predictor(Layer):
  """Layer that might need to pre-process the data, or have more complex state"""

  def preprocess_dataset(self, is_train, examples, dataset):
    pass

  def forward(self, features, label=None, **kwargs):
    raise NotImplementedError()

  def has_batch_loss(self):
    return False

  def get_state(self):
    return self.state_dict()

  def load_state(self, state):
    self.load_state_dict(state)


