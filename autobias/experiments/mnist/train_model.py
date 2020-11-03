import argparse
import logging
from os.path import join, basename
from time import perf_counter

from autobias.argmin_modules.affine_nll import AffineNLL
from autobias.argmin_modules.argmin_function import NumpyOptimizer
from autobias.argmin_modules.argmin_transform import ArgminTransformFunction
from autobias.argmin_modules.l2_norm import L2NormPenalty
from autobias.datasets.mnist import *
from autobias.experiments.train_args import add_train_args
from autobias.model.mnist_model import ImageClfModel, NullMNISTPredictor
from autobias.model.mnist_model import PredictFromFlattened, FromBiasFeaturePredictor
from autobias.modules.classifier_ensembles import ClfArgminEnsemble, ClfHead, \
  ClfBiAdversary
from autobias.modules.layers import seq, FullyConnected, Dropout, NullMapper, Conv2d
from autobias.training import train_utils
from autobias.training.data_batcher import SubsetSampler, StratifiedSampler
from autobias.training.data_iterator import TorchDataIterator
from autobias.training.evaluate import run_evaluation
from autobias.training.evaluator import ClfEnsembleEvaluator
from autobias.training.optimizer import SGD
from autobias.training.post_fit_rescaling_ensemble import FitRescaleParameters
from autobias.training.trainer import EvalDataset, Trainer, StoppingPoint
from autobias.utils import py_utils

"""
Run our MNIST experiments, this models are small enough it is seems best to train them
on a single CPU core. Multiple experiments can be run in parralell by using this 
script with the `init_only` flag, then running `training_models_batch` on the 
newly created output directory.
"""


def get_low_capacity_model(sz=28, n_classes=10):
  return PredictFromFlattened(
    NullMapper(),
    seq(
      FullyConnected(sz * sz * 3, 128, "relu"),
      Dropout(0.5),
      FullyConnected(128, n_classes, None)
    )
  )


def get_high_capacity_model(sz=28, n_classes=10):
  return PredictFromFlattened(
    Conv2d(3, 8, (7, 7)),
    seq(
      FullyConnected(8 * (sz - 6) * (sz - 6), 128, "relu"),
      Dropout(0.5),
      FullyConnected(128, n_classes, None)
    )
  )


MODES = [
  "none", "mce", "oracle", "nobp", "adv", "noci",
]


def main():
  py_utils.add_stdout_logger()
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", choices=["patch", "split", "background"], required=True,
                      help="Bias to train on")
  add_train_args(parser, entropy_w=False, default_adv_penalty=None, default_batch_size=1024,
                 default_epochs=100, default_entropy_penalty=None, lc_weight_default=None)

  parser.add_argument("--lr", type=float, default=0.01)
  parser.add_argument("--nruns", type=int, default=1)

  args = parser.parse_args()

  dataset = args.dataset

  if dataset == "patch":
    ds = MNISTPatches
    n_classes = 10
    w = 30
  elif dataset == "background":
    ds = MNISTBackgroundColor
    w = 28
    n_classes = 10
  elif dataset == "split":
    ds = MNISTDependent
    n_classes = 4
    w = 30
  else:
    raise NotImplementedError(f"Unknown dataset {dataset}")

  p = 0.9
  n_per_class = 200
  train = ds(p, True, (0, n_per_class))

  opt = SGD(args.lr, momentum=0.9)

  eval_sets = [
    EvalDataset(ds(p, True, (1400, 2400)),
                TorchDataIterator(SubsetSampler(None, args.batch_size)), "id"),
    EvalDataset(ds(1./n_classes, True, (1400, 2400)),
                TorchDataIterator(SubsetSampler(None, args.batch_size)), "od"),
  ]

  train.cache = True
  for ds in eval_sets:
    ds.cache = True

  def build_model():
    hc = get_high_capacity_model(w, n_classes)
    if args.mode == "none":
      # An ensemble with a Null predictor
      predictor = ClfArgminEnsemble(
        [
          ClfHead(predictor=NullMNISTPredictor(n_classes), head_name="bias"),
          ClfHead(predictor=hc, head_name="debiased")
        ],
        n_classes,
      )
    elif args.mode == "adv":
      if args.adversary_loss is None:
        if dataset == "patch":
          adv_loss = 0.01
        elif dataset == "background":
          adv_loss = 0.08
        elif dataset == "split":
          adv_loss = 0.01
        else:
          raise RuntimeError()
      else:
        adv_loss = args.adversary_loss

      if args.lc_weight is None:
        # Default depends on the bias
        if dataset == "patch":
          lc_w = 0.7
        elif dataset == "background":
          lc_w = 0.05
        elif dataset == "split":
          lc_w = 0.02
        else:
          raise RuntimeError()
      else:
        lc_w = args.lc_weight

      predictor = ClfBiAdversary(
        hc, get_low_capacity_model(w, n_classes),
        n_classes, adv_w=adv_loss, bias_loss=lc_w,
        main_loss=0.0, joint_loss=1.0, use_y_values=True, joint_adv=False
      )
    elif args.mode == "oracle":
      # An ensemble with a gold bias-predictor
      bias = FromBiasFeaturePredictor(p, n_classes)
      predictor = ClfArgminEnsemble(
        [
          ClfHead(predictor=bias, head_name="bias"),
          ClfHead(predictor=hc, head_name="debiased")
        ],
        n_classes,
      )
    else:
      if args.mode.startswith("mce"):
        rescaler = lambda: ArgminTransformFunction(AffineNLL(
          n_classes, n_classes, NumpyOptimizer(),
          residual=True, penalty=L2NormPenalty(0.002),
          fix_last_bias_to_zero=True,
        ))
      elif args.mode == "noci":
        rescaler = lambda: None
      elif args.mode == "nobp":
        rescaler = lambda: ArgminTransformFunction(AffineNLL(
          n_classes, n_classes, NumpyOptimizer(),
          residual=True, penalty=L2NormPenalty(0.002),
          fix_last_bias_to_zero=True,
        ), backprop_argmin=False)
      else:
        raise ValueError("Unknown mode: " + args.mode)

      predictor = ClfArgminEnsemble(
        [
          ClfHead(
            predictor=get_low_capacity_model(w, n_classes), head_name="bias",
            rescaler=rescaler(),
            nll_penalty=0.2 if args.lc_weight is None else args.lc_weight,
          ),
          ClfHead(
            predictor=hc, head_name="debiased",
            rescaler=rescaler(),
          )
        ],
        n_classes
      )

    return ImageClfModel(predictor)

  evaluator = ClfEnsembleEvaluator()

  if args.mode in {"mce", "nobp"}:
    hook = FitRescaleParameters(1024, None, sort=False)
  else:
    hook = None

  trainer = Trainer(
    opt,
    train,
    eval_sets,
    train_eval_iterator=TorchDataIterator(SubsetSampler(None, args.batch_size)),
    train_iterator=TorchDataIterator(
      StratifiedSampler(args.batch_size, n_repeat=10)),
    num_train_epochs=args.epochs,
    evaluator=evaluator,
    pre_eval_hook=hook,
    tb_factor=args.batch_size/256,
    save_each_epoch=False,
    progress_bar=True,
    eval_progress_bar=False,
    epoch_progress_bar=False,
    early_stopping_criteria=StoppingPoint("train", "nll/joint", 3e-4, 3),
    log_to_tb=False,
  )

  for r in range(args.nruns):
    if args.nruns > 1:
      print("")
      print("")
      print("*" * 10 + f" STARTING RUN {r+1}/{args.nruns} " + "*" * 10)

    # Build a model for each run to ensure it is fully reset
    model = build_model()

    if args.output_dir:
      if r == 0:
        train_utils.clear_if_nonempty(args.output_dir)
        train_utils.init_model_dir(args.output_dir, trainer, model)

      subdir = train_utils.select_subdir(args.output_dir)
    else:
      subdir = None

    if args.init_only:
      return

    if subdir is not None:
      logging.info(f"Start run for {subdir}")

    if args.time:
      t0 = perf_counter()
    else:
      t0 = None

    try:
      if subdir is not None:
        with open(join(subdir, "console.out"), "w") as f:
          trainer.training_run(model, subdir, no_cuda=True, print_out=f)
      else:
        trainer.training_run(model, subdir, no_cuda=True)
    except Exception as e:
      if args.nruns == 1 or isinstance(e, KeyboardInterrupt):
        raise e
      logging.warning("Error during training: " + str(e))
      continue

    if args.time:
      logging.info(f"Training took {perf_counter() - t0:.3f} seconds")


if __name__ == '__main__':
  main()
