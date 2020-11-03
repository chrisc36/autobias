import argparse
import logging
from collections import OrderedDict

import numpy as np
import torch

from autobias.datasets.imagenet_animals import ImageNetAnimals10k, load_imagenet_patch_bias
from autobias.training.evaluate import get_evaluation
from autobias.training.evaluator import Evaluator
from autobias.utils import py_utils, ops
from autobias.utils.py_utils import extract_models


class BiasThresholdEvaluator(Evaluator):
  """Use to evaluate ImagenetAnimals at different bias thresholds"""

  def __init__(self, thresholds):
    self.thresholds = thresholds
    self._bias_scores = []

    self._stats = {k: [] for k in ["debias", "bias", "joint"]}
    self._bias_annotations = {}
    self._class_weights = None

  def preprocess(self, datasets):
    # Load the biases
    total_per_label = np.zeros(6)
    for ds in datasets:
      bias = load_imagenet_patch_bias(ds)
      examples = ds.load()
      bias_map = {ex.example_id: ex.percents for ex in bias}
      mean_acc = np.mean([bias_map[ex.example_id][ex.label] for ex in examples])
      logging.info(f"Loaded annotations for {ds.fullname}, mean acc={mean_acc*100:.2f}")
      for ex in examples:
        total_per_label[ex.label] += 1
        score = bias_map[ex.example_id][ex.label]
        self._bias_annotations[ex.example_id] = score

    self._class_weights = total_per_label.mean() / total_per_label

  def clear(self):
    self._bias_scores = []
    self._stats = {k: [] for k in ["debias", "bias", "joint"]}

  def get_stats(self):
    bias_scores = np.array(self._bias_scores)
    bias_scores, weights = bias_scores[:, 0], bias_scores[:, 1]

    out = OrderedDict()
    for k in ["debias", "bias", "joint"]:
      v = np.concatenate(self._stats[k], 0)
      for thresh in self.thresholds:
        keep = bias_scores <= thresh
        w_keep = weights[keep]
        out[k + "/" + str(thresh)] = np.dot(v[keep], w_keep) / w_keep.sum()

        if thresh < 1.0:
          keep = bias_scores > thresh
          w_keep = weights[keep]
          out[k + "/-" + str(thresh)] = np.dot(v[keep], w_keep) / w_keep.sum()

    return out

  def evaluate_batch(self, examples, predictions, labels):
    for ex in examples:
      self._bias_scores.append((self._bias_annotations[ex.example_id], self._class_weights[ex.label]))

    logits = predictions.logprobs
    bias = logits[:, :, 0]
    debiased = logits[:, :, 1]
    self._stats["debias"].append(ops.numpy(torch.argmax(debiased, 1) == labels))
    self._stats["bias"].append(ops.numpy(torch.argmax(bias, 1) == labels))
    self._stats["joint"].append(ops.numpy(torch.argmax(debiased + bias, 1) == labels))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", nargs="+")
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--n_workers", type=int, default=12)
  parser.add_argument("--test", action="store_true")
  args = parser.parse_args()

  py_utils.add_stdout_logger()
  if args.test:
    ds = ImageNetAnimals10k("test", None)
  else:
    ds = ImageNetAnimals10k("dev", None)

  evaluator = BiasThresholdEvaluator([0.05, 0.1, 0.15, 0.2, 0.3, 0.8, 1.0])
  evaluator_name = "text-thresh-eval-v2"
  evaluator.preprocess([ds])

  models = extract_models(args.model)
  if sum(len(runs) for model_dir, runs in models.values()) == 0:
    print("No models selected")
    return

  get_evaluation(
    models, args.nocache, ds, evaluator, evaluator_name,
    128, sort=False, n_workers=args.n_workers)



if __name__ == '__main__':
  main()