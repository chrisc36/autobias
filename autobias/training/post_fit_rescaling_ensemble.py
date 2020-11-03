import torch

from autobias.model.data_parallel import DataParallel
from autobias.modules.classifier_ensembles import ClfArgminEnsemble
from autobias.training.data_batcher import SubsetSampler
from autobias.training.data_iterator import TorchDataIterator
from autobias.training.trainer import PreEvalHook
from autobias.utils import py_utils


class FitRescaleParameters(PreEvalHook):
  """
  Uses the model to compute features for examples in the training data, then
  feeds those features into the model for it to use to decide what
  values argmin values to use during evaluation
  """
  def __init__(self, batch_size, in_progress_sample, eval_mode=False, sort=True):
    self.batch_size = batch_size
    self.in_progress_sample = in_progress_sample
    self.eval_mode = eval_mode
    self.sort = sort
    self._iterator = None
    self.split_batch = None

  def init(self, training_examples, collate_fn, split_batch=None):
    self.split_batch = split_batch
    iterator = TorchDataIterator(SubsetSampler(self.in_progress_sample, self.batch_size, self.sort))
    self._iterator = iterator.build_iterator(training_examples, collate_fn, split_batch)

  def post_fit(self, is_final, model, batch_to_device):
    if isinstance(model, DataParallel):
      predictor: ClfArgminEnsemble = model.model.predictor
    else:
      predictor: ClfArgminEnsemble = model.predictor

    if not hasattr(predictor, "needs_eval_features") or not predictor.needs_eval_features():
      return

    if self.eval_mode:
      model.eval()
    else:
      model.train()

    all_features = []
    all_labels = []
    device = next(predictor.parameters()).device

    for batch in self._iterator:
      batch = batch_to_device(batch)
      with torch.no_grad():
        features = model(*batch, mode="features")
        if self.split_batch:
          all_labels += [x[1].to(device) for x in batch]
        else:
          all_labels.append(batch[1])
        all_features.append(features)

    per_rescaler_features = py_utils.transpose_lists(all_features)
    labels = torch.cat(all_labels, 0)
    predictor.set_eval_features(per_rescaler_features, labels)

