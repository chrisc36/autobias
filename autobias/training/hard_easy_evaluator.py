import logging
from collections import OrderedDict, defaultdict
from os.path import exists, join

import numpy as np

from autobias import config
from autobias.datasets.entailment_datasets import load_hypothesis_bias
from autobias.modules.classifier_ensembles import ClassifierEnsembleOutput
from autobias.training.evaluator import Evaluator
from autobias.utils import py_utils, ops, downloader


class ClfHardEasyEvaluator(Evaluator):
  def __init__(self, accuracy=True, prob=False, nll=True,
               prefix_format="{split}-{output}-{metric}", per_n_class=None,
               eval_joint=True):
    super().__init__()
    self.eval_joint = eval_joint
    self.per_n_class = per_n_class
    self.accuracy = accuracy
    self.prob = prob
    self.nll = nll
    self.prefix_format = prefix_format
    self._annotations = {}

    if self.per_n_class is not None:
      self._stats = defaultdict(lambda: np.zeros(self.per_n_class))
      self._split_to_sum = defaultdict(lambda: np.zeros(per_n_class))
    else:
      self._stats = defaultdict(float)
      self._split_to_sum = defaultdict(float)

  def clear(self):
    if self.per_n_class is not None:
      self._stats = defaultdict(lambda: np.zeros(self.per_n_class))
      self._split_to_sum = defaultdict(lambda: np.zeros(self.per_n_class))
    else:
      self._stats = defaultdict(float)
      self._split_to_sum = defaultdict(float)

  def get_stats(self):
    keys = sorted(self._stats.keys())
    out = OrderedDict()
    for key in keys:
      output_name, split, metric = key
      stat = self._stats[key] / self._split_to_sum[split]
      if self.per_n_class:
        stat = stat.mean()
      out[self.prefix_format.format(split=split, metric=metric, output=output_name)] = stat
    return out

  def preprocess(self, datasets):
    for ds in datasets:
      logging.info("Loading hard/easy annotation for %s" % ds.fullname)
      data, model_out = load_hypothesis_bias(ds.fullname)
      labels = {x.example_id: x.label for x in ds.load()}
      labels = np.array([labels[x] for x in data])

      model_pred = np.argmax(model_out.logprobs, 1)
      is_correct = (model_pred == labels)
      n_correct = is_correct.sum()
      n_incorrect = len(is_correct) - n_correct
      target_acc = 1 / 3.0
      correct_ratio = target_acc * n_incorrect / (n_correct - target_acc * n_correct)
      weights = (is_correct * correct_ratio + (1 - is_correct))
      for example_id, w, c in zip(data, weights, is_correct):
        if example_id in self._annotations:
          raise ValueError(example_id)
        self._annotations[example_id] = w, 1 - c

  def evaluate_batch(self, examples, output, labels):
    labels = np.array([x.label for x in examples])
    ix = np.arange(len(labels))

    weights, is_hard = py_utils.transpose_lists(self._annotations[x.example_id] for x in examples)
    weights = np.array(weights, dtype=np.double)
    unweighted = np.ones_like(weights)
    hard = np.array(is_hard, dtype=np.double)

    if isinstance(output, ClassifierEnsembleOutput):
      outputs = []
      base_or_zero = 0 if output.base is None else output.base
      for i, h in enumerate(output.head_names):
        outputs.append((h, output.logprobs[:, :, i] + base_or_zero))
      if self.eval_joint:
        outputs.append(("joint", output.logprobs.sum(-1) + base_or_zero))
    else:
      outputs = [("joint", output.logprobs)]

    def _add_vec_to_annotations(key, vec, _w):
      if self.per_n_class:
        np.add.at(self._stats[key], labels, _w * vec)
      else:
        self._stats[key] += _w.dot(vec)

    outputs = [(k, ops.numpy(v)) for k, v in outputs]
    for split, w in [("ind", unweighted), ("ood", hard)]:
      if self.per_n_class:
        np.add.at(self._split_to_sum[split], labels, w)
      else:
        self._split_to_sum[split] += w.sum()

      for output_name, logits in outputs:
        if self.accuracy:
          _add_vec_to_annotations((output_name, split, "acc"), np.argmax(logits, 1) == labels, w)
        if self.prob:
          _add_vec_to_annotations((output_name, split, "prob"), np.exp(logits[ix, labels]), w)
        if self.nll:
          _add_vec_to_annotations((output_name, split, "nll"), logits[ix, labels], w)

