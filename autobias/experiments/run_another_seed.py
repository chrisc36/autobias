import argparse
import logging
import warnings
from os.path import join

from autobias.training import train_utils

warnings.simplefilter(action='ignore', category=FutureWarning)

from autobias.model.model import Model
from autobias.training.trainer import Trainer
from autobias.utils import load_config, py_utils


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("output_dir", nargs="+")
  parser.add_argument("--n_processes", type=int, default=4)
  parser.add_argument("--fp16", action="store_true")
  parser.add_argument("--seed", type=int)
  parser.add_argument("--nruns", type=int, default=1)
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  for src in args.output_dir:
    output_dir = src
    for i in range(args.nruns):
      trainer: Trainer = load_config.load_config(join(output_dir, "trainer.json"))
      subdir = train_utils.select_subdir(src)
      logging.info(f"Starting run for directory {subdir}")
      model: Model = load_config.load_config(join(output_dir, "model.json"))
      trainer.training_run(model, subdir, args.seed, args.n_processes, fp16=args.fp16)


if __name__ == "__main__":
  main()
