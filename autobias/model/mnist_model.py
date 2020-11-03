from typing import List

import torch
from PIL import Image
from torchvision import transforms

from autobias.datasets.dataset import Dataset
from autobias.model.model import Model, Predictor
from autobias.utils import ops


class PredictFromFlattened(Predictor):
  """Predict using the flattened features"""

  def __init__(self, base, fc):
    super().__init__()
    self.base = base
    self.fc = fc

  def reset_parameters(self):
    pass

  def forward(self, x, label=None, **kwargs):
    embed = self.base(x[0])
    post = torch.flatten(embed, 1)
    return self.fc(post)


class FromBiasFeaturePredictor(Predictor):
  """Predict using the bias feature"""

  def __init__(self, p, n_classes):
    super().__init__()
    self.p = p
    self.n_classes = n_classes

  def reset_parameters(self):
    pass

  def forward(self, features, label=None, **kwargs):
    bias = features[1]["bias"]
    out = torch.log(ops.smooth_one_hot(bias, self.n_classes, 1.0 - self.p))
    return out


class NullMNISTPredictor(Predictor):
  """Always predicts a uniform distribution"""

  def __init__(self, n_out):
    self.n_out = n_out
    super().__init__()

  def reset_parameters(self):
      pass

  def forward(self, x, label=None, **kwargs):
    x = x[0]
    return torch.zeros(len(x), self.n_out, device=x.device)


class ImageClfModel(Model):
  def __init__(self, predictor, train_transform=None, eval_transform=None):
    super().__init__()
    self.train_transform = train_transform
    self.eval_transform = eval_transform
    self.predictor = predictor

  def preprocess_train(self, train: Dataset, other: List[Dataset], n_processes=None):
    return self._preprocess_datasets(True, [train] + other, n_processes)

  def preprocess_datasets(self, datasets: List[Dataset], n_processes=None):
    return self._preprocess_datasets(False, datasets, n_processes)

  def _preprocess_datasets(self, first_is_train, datasets: List[Dataset], n_processes=None):
    data = [x.load() for x in datasets]

    if self.train_transform is None and self.eval_transform is None:
      to_tensor = transforms.ToTensor()
    else:
      to_tensor = None

    for i, (dataset, examples) in enumerate(zip(datasets, data)):
      if to_tensor is not None and not isinstance(examples[0].image, torch.Tensor):
        for ex in examples:
          ex.image = to_tensor(ex.image)
      self.predictor.preprocess_dataset(i == 0 and first_is_train, examples, dataset)
    return data

  def get_collate_fn(self):
    pre = self.eval_transform

    def collate(batch: List[ImageClfModel]):
      if pre is not None:
        images = [pre(ex.image.convert('RGB')) for ex in batch]
      else:
        images = [x.image for x in batch]
      images = torch.stack(images, 0)
      labels = torch.tensor([x.label for x in batch], dtype=torch.long)
      if batch[0].other_features is not None:
        other_features = ops.collate_flat_dict([x.other_features for x in batch])
      else:
        other_features = {}
      return (images, other_features), labels
    return collate

  def get_collate_train_fn(self):
    if self.train_transform is None:
      return self.get_collate_fn()

    def collate(batch: List[ImageClfModel]):
      images = torch.stack([self.train_transform(Image.open(ex.image).convert('RGB')) for ex in batch], 0)
      labels = torch.tensor([x.label for x in batch], dtype=torch.long)
      if batch[0].other_features is not None:
        other_features = ops.collate_flat_dict([x.other_features for x in batch])
      else:
        other_features = {}
      return (images, other_features), labels
    return collate

  def get_state(self):
    return self.predictor.get_state()

  def load_state(self, state):
    self.predictor.load_state(state)

  def forward(self, features, label=None, **kwargs):
    return self.predictor(features, label, **kwargs)


