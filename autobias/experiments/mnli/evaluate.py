import argparse
import logging

from autobias.datasets.entailment_datasets import MnliDevMatched, MnliDevUnmatched, Hans
from autobias.training.evaluate import run_evaluation, \
  get_cached_evaluations
from autobias.training.evaluator import ClfEnsembleEvaluator
from autobias.utils import py_utils, load_word_vectors
from autobias.utils.load_config import ClfHardEasyEvaluator
from autobias.utils.py_utils import extract_models


def main():
  # Re-use if we are evaluating multiple models
  load_word_vectors.GLOBAL_CACHE = {}

  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir", nargs="+")
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--dataset", choices=["hans", "dev", "test"], default="dev")
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.dataset == "test":
    ds = MnliDevUnmatched()
  elif args.dataset == "dev":
    ds = MnliDevMatched()
  else:
    ds = Hans()
  ds.cache = True
  models = extract_models(args.output_dir)

  if len(models) == 0:
    print("No models found")
    return

  if args.dataset == "hans":
    evaluator = ClfEnsembleEvaluator(output_format="{output}-{metric}")
    evaluator_name = "ensemble-eval-v1"
  else:
    evaluator = ClfHardEasyEvaluator(prefix_format="{output}-{metric}/{split}")
    evaluator_name = "hard-easy-eval-v1"

  models, all_stats = get_cached_evaluations(models, ds, evaluator_name)

  if len(models) == 0:
    print("All models were cached")
    return
  elif len(all_stats) == 0:
    logging.info("No models were cached")
  else:
    logging.info(f"{len(all_stats)} models ({sum(len(x) for x in all_stats.items())}) were cached")

  evaluator.preprocess([ds])
  for name, (model_dir, runs) in models.items():
    logging.info(f"Evaluating model: {name} ({len(runs)} runs)")
    for run in runs:
      run_evaluation(run, ds, args.batch_size, evaluator, evaluator_name,
                     cache=True, cache_model_output=True, n_processes=4)


if __name__ == '__main__':
  main()
