import argparse

from autobias.argmin_modules.affine_nll import AffineNLL
from autobias.argmin_modules.argmin_function import NumpyOptimizer
from autobias.argmin_modules.argmin_transform import ArgminTransformFunction
from autobias.argmin_modules.l2_norm import L2NormPenalty
from autobias.datasets.imagenet_animals import ImageNetAnimals10k
from autobias.experiments.train_args import add_train_args
from autobias.model.resent_model import ResnetPredictor, ResNetModel, FromResnetPredictor, \
  NullResnetPredictor, ResnetOracleBias, ExtractLastEmbeddings
from autobias.modules.classifier_ensembles import ClfHead, ClfArgminEnsemble, ClfBiAdversary, \
  ClfBiasMixinEnsemble
from autobias.modules.image_transforms import ImageNetEvalTransform, RandomCropTransform
from autobias.modules.layers import seq, FullyConnected, Conv1x1, NullMapper, \
  AveragePooling2d
from autobias.training import train_utils
from autobias.training.data_batcher import SubsetSampler, StratifiedSampler
from autobias.training.data_iterator import TorchDataIterator
from autobias.training.evaluator import ClfEnsembleEvaluator
from autobias.training.optimizer import SGD, PiecewiseLinear
from autobias.training.post_fit_rescaling_ensemble import FitRescaleParameters
from autobias.training.trainer import Trainer, EvalDataset
from autobias.utils import py_utils


def get_high_capacity():
  return ResnetPredictor(512, 6, return_logits=True)


def get_low_capacity():
  return FromResnetPredictor(
    seq(
      Conv1x1("relu", 64, 256),
      NullMapper(),
      AveragePooling2d(),
      seq(
        FullyConnected(256, 256),
        FullyConnected(256, 6, fn=None)
      )
    ),
    1,
    return_logits=True,
  )


def main(args=None, init_only=False):
  py_utils.add_stdout_logger()
  parser = argparse.ArgumentParser()
  add_train_args(parser, default_entropy_penalty=0.1, default_adv_penalty=0.005,
                 default_epochs=38, default_batch_size=512, lc_weight_default=None)
  parser.add_argument("--n_workers", type=int, default=12,
                      help="N workers to use when loading images")
  args = parser.parse_args(args)

  if args.lc_weight is None:
    if args.mode == "adv":
      lc_weight = 0.3
    else:
      lc_weight = 0.2
  else:
    lc_weight = args.lc_weight

  dbg = args.debug
  n_classes = 6

  if args.mode in {"mce", "noci"}:
    if args.mode == "noci":
      rescaler = lambda: None
    else:
      rescaler = lambda: ArgminTransformFunction(AffineNLL(
        n_classes, n_classes, NumpyOptimizer(),
        residual=True, penalty=L2NormPenalty(0.002),
        fix_last_bias_to_zero=True,
      ), backprop_argmin=args.mode != "nobp")

    predictor = ClfArgminEnsemble([
      ClfHead(
        get_low_capacity(), head_name="bias",
        rescaler=rescaler(), nll_penalty=lc_weight
      ),
      ClfHead(
        get_high_capacity(), head_name="debiased",
        rescaler=rescaler(),
      )
    ], n_classes, add_prior=False)

  elif args.mode == "none":
    predictor = ClfArgminEnsemble([
      ClfHead(NullResnetPredictor(6), head_name="bias", rescaler=None),
      ClfHead(get_high_capacity(), head_name="debiased", rescaler=None)
    ], n_classes, add_prior=False)
  elif args.mode == "oracle":
    predictor = ClfBiasMixinEnsemble(
      ExtractLastEmbeddings(),
      6,
      ResnetOracleBias(),
      512,
      args.entropy_penalty
    )
  elif args.mode == "adv":
    predictor = ClfBiAdversary(
      get_high_capacity(), get_low_capacity(), n_classes, args.adversary_loss,
      joint_loss=1.0,
      bias_loss=lc_weight, use_y_values=True,
      joint_adv=False,
    )
  else:
    raise RuntimeError()

  evaluator = ClfEnsembleEvaluator()

  model = ResNetModel(
    predictor, arch="resnet18", from_pretrained=False,
    resize=256,
    eval_transform=ImageNetEvalTransform(224, resize=False),
    train_transform=RandomCropTransform(224),
  )

  opt = SGD(0.02, momentum=0.9, schedule=PiecewiseLinear([args.epochs - 5], 0.3))

  num_workers = args.n_workers
  n_train_workers = num_workers
  test_batch_size = 512

  train = ImageNetAnimals10k("train", 300 if args.debug else None)
  dev = ImageNetAnimals10k("dev", 150 if args.debug else None)

  eval_sets = [
    EvalDataset(
      dev,
      TorchDataIterator(
        SubsetSampler(None if args.debug else 12000, test_batch_size),
        pin_memory=True, num_workers=num_workers), "dev"),
  ]

  trainer = Trainer(
    opt,
    train,
    eval_sets,
    train_eval_iterator=TorchDataIterator(SubsetSampler(None if args.debug else 8000, args.batch_size),
                                          num_workers=n_train_workers, pin_memory=True),
    train_iterator=TorchDataIterator(
      StratifiedSampler(test_batch_size, n_repeat=2), pin_memory=True, num_workers=n_train_workers),
    num_train_epochs=3 if dbg else args.epochs,
    evaluator=evaluator,
    tb_factor=args.batch_size/256,
    pre_eval_hook=FitRescaleParameters(test_batch_size, None if args.debug else 4096, sort=False),
    save_best_model=("dev", "acc/joint"),
    eval_on_epochs=2,
    split_then_collate=True
  )

  if init_only or args.init_only:
    train_utils.init_model_dir(args.output_dir, trainer, model)
  else:
    trainer.train(model, args.output_dir, args.seed, args.n_processes, fp16=args.fp16, no_cuda=args.nocuda)


if __name__ == "__main__":
  main()
