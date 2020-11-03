import argparse
import logging
import warnings
from os import environ
from os.path import join, basename
from typing import Iterable

from autobias.training import train_utils
from autobias.training.evaluate import run_evaluation
from autobias.utils.process_par import Processor, process_par

warnings.simplefilter(action='ignore', category=FutureWarning)

from autobias.model.model import Model
from autobias.training.trainer import Trainer
from autobias.utils import py_utils, load_config


class MnistCpuTrainer(Processor):
  def process(self, data: Iterable):
    for output_dir in data:
      trainer: Trainer = load_config.load_config(join(output_dir, "trainer.json"))
      eval_sets = trainer.eval_sets
      trainer.eval_sets = []
      model: Model = load_config.load_config(join(output_dir, "model.json"))
      subdir = train_utils.select_subdir(output_dir)
      seed = int(basename(subdir)[1:])
      trainer.progress_bar = False
      trainer.eval_progress_bar = False
      with open(join(subdir, "console.out"), "w") as f:
        try:
          trainer.training_run(
            model, subdir, seed, None, fp16=False, print_out=f, no_cuda=True)
        except Exception as e:
          if isinstance(e, KeyboardInterrupt):
            raise e
          logging.warning("Error during training: " + str(e))
          continue

      logging.info("Evaluating...")
      evaluator = trainer.evaluator
      for ds in [trainer.train_dataset, eval_sets[0].dataset, eval_sets[1].dataset]:
        run_evaluation(subdir, ds, 512, evaluator, "ensemble-eval-v1",
                       cache=True, model=model, sort=False, progress_bar=False)
    return []


def set_to_single_thread():
  # One these will work, probably
  environ["OMP_NUM_THREADS"] = "1"
  environ["MKL_NUM_THREADS"] = "1"
  environ["NUMEXPR_NUM_THREADS"] = "1"
  environ["OPENBLAS_NUM_THREADS"] = "1"


def main():
  parser = argparse.ArgumentParser(description="Train many MNIST models in parallel.")
  parser.add_argument("output_dir", nargs="+")
  parser.add_argument("--nosingle_thread", help="By default, we restrict oursevles to one thread"
                                                "per a process. In my experience that makes traiing"
                                                "multiple models much faster. But this flag"
                                                "turn that mode off.")
  parser.add_argument("--n_processes", type=int, required=True)
  parser.add_argument("--nruns", type=int, default=1)
  args = parser.parse_args()

  if args.nosingle_thread:
    init = None
  else:
    init = set_to_single_thread

  models = py_utils.extract_models(args.output_dir, require_runs=False)
  total_left = 0
  output_dir_to_count = {}
  for model_dir, runs in models.values():
    n_left = max(args.nruns - len(runs), 0)
    print(f"{model_dir} has {n_left} remaining")
    total_left += n_left
    output_dir_to_count[model_dir] = n_left

  print(f"Queueing all {total_left} runs...")
  targets = py_utils.flatten_list([k] * v for k, v in output_dir_to_count.items())
  process = MnistCpuTrainer()
  print()
  process_par(targets, process, args.n_processes, chunk_size=1, initializer=init)


if __name__ == "__main__":
  main()