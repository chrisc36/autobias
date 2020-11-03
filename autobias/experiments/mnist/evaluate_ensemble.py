import argparse
import json
import logging
from os import path
from os.path import join, exists, normpath

from autobias.datasets.mnist import MNISTBackgroundColor, MNISTPatches, MNISTDependent
from autobias.training.evaluate import get_evaluation
from autobias.training.evaluator import ClfEnsembleEvaluator
from autobias.utils import py_utils


def get_background(is_test, in_domain):
  p = 0.9 if in_domain else 0.1
  c = (400, 1400) if is_test else (1400, 2400)
  return MNISTBackgroundColor(p, True, c)


def get_patch(is_test, in_domain):
  p = 0.9 if in_domain else 0.1
  c = (400, 1400) if is_test else (1400, 2400)
  return MNISTPatches(p, True, c)


def get_split(is_test, in_domain):
  p = 0.9 if in_domain else 1.0 / 4.0
  c = (400, 1400) if is_test else (1400, 2400)
  return MNISTDependent(p, True, c)


datasets_fns = {
  "background": get_background,
  "patch": get_patch,
  "split": get_split
}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", help="Directory of models of evaluate")
  parser.add_argument("--dataset", choices=list(datasets_fns.keys()),
                      help="Dataset to evaluate on, if not set if will be inferred"
                           " based on the what the models were trained on")
  parser.add_argument("--nocache", action="store_true")
  parser.add_argument("--test", action="store_true", help="Evaluated on the test data")
  args = parser.parse_args()

  is_test = args.test

  py_utils.add_stdout_logger()
  models = py_utils.extract_models(args.model)
  if len(models) == 0:
    logging.info("No models found")
    return 0

  if args.dataset is not None:
    fn = datasets_fns[args.dataset]
  else:
    all_train_ds = None
    for model_dir, _ in models.values():
      trainer = json.load(open(join(model_dir, "trainer.json")))
      train_ds = trainer["train_dataset"]["name"]
      if all_train_ds is None:
        all_train_ds = train_ds
      elif all_train_ds != train_ds:
        raise ValueError("No dataset given, and unable to infer seems "
                         "models were trained on different datasets")
    logging.info(f"All models were trained on {all_train_ds}, so testing on the same bias")
    if all_train_ds == "MNISTPatches":
      fn = get_patch
    elif all_train_ds == "MNISTDependent":
      fn = get_split
    elif all_train_ds == "MNISTBackgroundColor":
      fn = get_background
    else:
      raise ValueError()

  id_test, ood_test = fn(is_test, True), fn(is_test, False)

  evaluator = ClfEnsembleEvaluator()

  logging.info("Evaluating OOD Test")
  get_evaluation(
    models, args.nocache, ood_test, evaluator, "ensemble-eval-v1", 128, sort=False,
    progress_bar=False
  )

  logging.info("Evaluating ID Test")
  get_evaluation(
    models, args.nocache, id_test, evaluator, "ensemble-eval-v1", 128, sort=False,
    progress_bar=False
  )


if __name__ == '__main__':
  main()
