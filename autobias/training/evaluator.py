from collections import OrderedDict, Counter
from typing import Union

import numpy as np
import torch
from torch.nn import functional as F

from autobias.modules.classifier_ensembles import ClassifierOutput, ClassifierEnsembleOutput
from autobias.utils import ops
from autobias.utils.configured import Configured


class Evaluator(Configured):
  """Evalutors get fed a series of batches and with model predictions, during which some
  statistics are accumulated and can be retrieved with `get_stats`"""

  def clear(self):
    raise NotImplementedError()

  def get_stats(self):
    raise NotImplementedError()

  def evaluate_batch(self, examples, predictions, labels):
    raise NotImplementedError()

  def preprocess(self, datasets):
    """
    Give the evaluator a chance see that data before it is used, we need this
    since `ClfHardEasyEvaluator` needs to pre-load some dataset statistics before use
    """
    pass


class EvaluatorCounter(Evaluator):
  """For evaluators that produce dictionaries of scalar, (scalar, count) pairs"""

  def __init__(self):
    self._stats = OrderedDict()
    self._n = 0

  def clear(self):
    self._stats = OrderedDict()
    self._n = 0

  def get_stats(self):
    out = Counter()
    for k, v in self._stats.items():
      if isinstance(v, tuple):
        out[k] = v[0] / v[1]
      else:
        out[k] = v / self._n
    return out

  def evaluate_batch(self, examples, predictions, labels):
    for k, v in self.get_batch_stats(examples, predictions, labels).items():
      if isinstance(v, torch.Tensor):
        v = ops.numpy(v)
      cur = self._stats.get(k)
      if cur is None:
        self._stats[k] = v
      else:
        self._stats[k] += v
    self._n += len(examples)

  def get_batch_stats(self, examples, predictions, labels):
    raise NotImplementedError()


class ClfEvaluator(EvaluatorCounter):

  def __init__(self, accuracy=True, prob=False, nll=True, prefix_format="{metric}"):
    super().__init__()
    self.accuracy = accuracy
    self.prob = prob
    self.nll = nll
    self.prefix_format = prefix_format

  def get_batch_stats(self, examples, model_output, labels):
    logits = model_output.logprobs
    ix = np.arange(len(logits))
    result = OrderedDict()
    if self.accuracy:
      prefix = self.prefix_format.format(metric="acc")
      result[prefix] = ops.numpy((torch.argmax(logits, 1) == labels).sum())
    if self.prob:
      prefix = self.prefix_format.format(metric="prob")
      result[prefix] = ops.numpy(torch.exp(logits[ix, labels]).sum())
    if self.nll:
      prefix = self.prefix_format.format(metric="nll")
      result[prefix] = ops.numpy(-logits[ix, labels].sum())
    return result


class ClfEnsembleEvaluator(EvaluatorCounter):
  def __init__(self, output_format="{metric}/{output}", h1_name="bias", h2_name="debiased"):
    super().__init__()
    self.output_format = output_format
    self.h1_name = h1_name
    self.h2_name = h2_name

  def get_batch_stats(self, examples, predictions: ClassifierEnsembleOutput, labels):

    logits = predictions.logprobs
    bias = logits[:, :, 0]
    main = logits[:, :, 1]

    if predictions.base is not None:
      bias = bias + predictions.base
      debiased = main + predictions.base
      joint = main + bias + predictions.base
    else:
      debiased = main
      joint = main + bias

    heads = [
      (self.h1_name, bias),
      (self.h2_name, debiased),
      ("joint", joint)
    ]

    heads = heads[::-1]
    out = OrderedDict()

    for metric in ["acc", "nll"]:
      for name, logits in heads:
        if metric == "acc":
          score = torch.eq(torch.argmax(logits, 1), labels).float().sum()
        else:
          score = F.cross_entropy(logits, labels, reduction="sum")
        out[self.output_format.format(output=name, metric=metric)] = score

    return out

