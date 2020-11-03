from os.path import join, exists
from typing import List, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from autobias.config import IMAGENET_BIAS_CACHE
from autobias.datasets.dataset import Dataset
from autobias.datasets.image_clf_example import ImageClfExample
from autobias.datasets.imagenet_animals import load_imagenet_patch_bias
from autobias.model.model import Model, Predictor, TrainOutput
from autobias.modules import resnet
from autobias.modules.classifier_ensembles import ClassifierOutput
from autobias.modules.layers import Layer
from autobias.utils import ops
from autobias.utils.downloader import download_from_drive
from autobias.utils.load_images import load_resized_img_with_cache
from autobias.utils.process_par import Processor


class LearnedImageFeature:
  """Output of the resnet backbone"""

  def __init__(self, backbone_layers, other_features):
    self.backbone_layers = backbone_layers
    self.other_features = other_features


class ExtractLastEmbeddings(Layer):

  def reset_parameters(self):
    pass

  def forward(self, features: LearnedImageFeature, labels=None, **kwargs):
    return features.backbone_layers[-1]


class ResnetOracleBias(Predictor):
  """Predict using the bias"""

  def __init__(self, smooth_init=-3, fixed_smoothing=False):
    super().__init__()
    self.fixed_smoothing = fixed_smoothing
    self.smooth_init = smooth_init
    self._annotations = {}
    self.smooth = nn.Parameter(torch.full((1, ), self.smooth_init),
                               requires_grad=not self.fixed_smoothing)

  def reset_parameters(self):
    self.smooth.data.fill_(self.smooth_init)

  def preprocess_dataset(self, is_train, examples: List[ImageClfExample], dataset):
    bias = load_imagenet_patch_bias(dataset)
    percents = {x.example_id: x.percents for x in bias}
    for ex in examples:
      if ex.other_features is None:
        ex.other_features = {}
      ex.other_features["pretrained"] = percents[ex.example_id]

  def forward(self, features: LearnedImageFeature, label=None, **kwargs):
    pre = features.other_features["pretrained"]
    return torch.log(pre + F.softplus(self.smooth))


class ResnetPredictor(Predictor):
  """Predict using the last features from the backbone"""

  def __init__(self, n_out, n_classes, return_logits=False, mapper=None):
    super().__init__()
    self.mapper = mapper
    self.n_classes = n_classes
    self.return_logits = return_logits
    self.n_out = n_out
    self.clf = nn.Linear(n_out, n_classes)

  def reset_parameters(self):
    self.clf.reset_parameters()

  def forward(self, features: LearnedImageFeature, label=None, **kwargs):
    features = features.backbone_layers[-1]
    if self.mapper is not None:
      features = self.mapper(features)
    logits = self.clf(features)
    if self.return_logits:
      return logits
    if self.training:
      return TrainOutput(F.cross_entropy(logits, label))
    return ClassifierOutput(F.log_softmax(logits, -1))


class NullResnetPredictor(Predictor):
  """Always predicts a uniform distribution"""

  def __init__(self, n_out):
    self.n_out = n_out
    super().__init__()

  def reset_parameters(self):
      pass

  def forward(self, features, label=None, **kwargs):
    fe = features.backbone_layers[-1]
    return torch.zeros(len(fe), self.n_out, device=fe.device)


class FromResnetPredictor(Predictor):
  """Predict using the `nth` layer from the backbone"""

  def __init__(self, mapper, n_layer, return_logits=False):
    super().__init__()
    self.mapper = mapper
    self.return_logits = return_logits
    self.n_layer = n_layer

  def reset_parameters(self):
    pass

  def forward(self, features: LearnedImageFeature, label=None, **kwargs):
    logits = self.mapper(features.backbone_layers[self.n_layer])
    if self.return_logits:
      return logits
    if self.training:
      return TrainOutput(F.cross_entropy(logits, label))
    return ClassifierOutput(F.log_softmax(logits, -1))


class ResNetModel(Model):
  def __init__(self, predictor: Predictor, arch: str, eval_transform, from_pretrained=True,
               train_transform=None, n_layers=None, resize=None):
    super().__init__()
    self.resize = resize
    self.n_layers = n_layers
    self.from_pretrained = from_pretrained
    self.predictor = predictor
    self.eval_transform = eval_transform
    self.train_transform = train_transform
    self.arch = arch
    self.resnet = getattr(resnet, arch)(from_pretrained=from_pretrained, n_layers=n_layers)

  def preprocess_train(self, train: Dataset, other: List[Dataset], n_processes=None):
    return self._preprocess_datasets(True, [train] + other, n_processes)

  def preprocess_datasets(self, datasets: List[Dataset], n_processes=None):
    return self._preprocess_datasets(False, datasets, n_processes)

  def _preprocess_datasets(self, first_is_train, datasets: List[Dataset], n_processes=None):
    data = [x.load() for x in datasets]
    for i, (dataset, examples) in enumerate(zip(datasets, data)):
      self.predictor.preprocess_dataset(i == 0 and first_is_train, examples, dataset)
    return data

  def get_collate_fn(self):
    return self._get_collate_fn(False)

  def get_collate_train_fn(self):
    return self._get_collate_fn(True)

  def has_batch_loss(self):
    return self.predictor.has_batch_loss()

  def _get_collate_fn(self, is_train):
    if is_train and self.train_transform:
      transform = self.train_transform
    else:
      transform = self.eval_transform

    def collate(batch: List[ImageClfExample]):
      image_tensors = []
      for ex in batch:
        if not isinstance(ex.image, str):
          img = ex.image
        elif self.resize == 256:
          # Resizing is expensive, so we allow the resized images to be cached
          img = load_resized_img_with_cache(ex.image).convert('RGB')
        elif self.resize is None:
          img = Image.open(ex.image).convert('RGB')
        else:
          raise NotImplementedError()

        image_tensors.append(transform(img))

      images = torch.stack(image_tensors, 0)
      labels = torch.tensor([x.label for x in batch], dtype=torch.long)
      if batch[0].other_features is not None:
        other_features = ops.collate_flat_dict([x.other_features for x in batch])
      else:
        other_features = None
      return (images, other_features), labels
    return collate

  def forward(self, features, label=None, **kwargs):
    if self.predictor.has_batch_loss() and kwargs.get("mode") == "loss":
      return self.predictor(features, label, **kwargs)
    features, other_features = features
    fe = self.resnet(features)
    return self.predictor(LearnedImageFeature(fe, other_features), label, **kwargs)
