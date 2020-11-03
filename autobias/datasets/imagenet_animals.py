from collections import defaultdict
from os.path import exists, join
from typing import List

import numpy as np

from autobias.config import IMAGENET_HOME, IMAGENET_ANIMALS, IMAGENET_BIAS_CACHE
from autobias.datasets.dataset import Dataset
from autobias.datasets.image_clf_example import ImageClfExample
from autobias.utils import py_utils
from autobias.utils.downloader import download_from_drive

FILE_IDS = {
  "train": "1XWoxkQHEtC03TtWkHIkFttYoTZnq-X9j",
  "test": "1Hk0wc5eObHzbQowjYNOiS3U8UOCFNRpX",
  "dev": "1gWLxaHVYWIxa4bb8OEY0eBUU1S9cvE8w"
}


class ImageNetAnimals10k(Dataset):
  """
  Dataset for ImageNetAnimals, it return `ImageClfExample` with image ids indicating
  what imagenet image they should be paired with
  """
  def __init__(self, split, sample=None):
    src = join(IMAGENET_ANIMALS, "%s.tsv" % split)
    if not exists(src):
      download_from_drive(FILE_IDS[split], src)
    self.src_file = src
    self.sample = sample
    super().__init__("imagenet-animal-10k", split, None if sample is None else str(sample))

  def _load(self):
    per_class_examples = defaultdict(list)
    with open(self.src_file) as f:
      for line in f:
        image, class_ix = line.split("\t")
        class_ix = int(class_ix)
        example_id = image.split("/")[-1][:-5]
        per_class_examples[class_ix].append(ImageClfExample(example_id, join(IMAGENET_HOME, image), class_ix))

    if max(per_class_examples)+1 != len(per_class_examples):
      raise ValueError()

    out = []
    if self.sample:
      for class_ix, class_examples in per_class_examples.items():
        class_examples.sort(key=lambda x: x.example_id)
        np.random.RandomState(class_ix).shuffle(class_examples)
        out += class_examples[:self.sample]
    else:
      out = py_utils.flatten_list(per_class_examples.values())

    if len(out) == 0:
      raise ValueError("No examples loaded")
    return out


IMAGENET_BIAS_FILEIDS = {
  "dev": "1K8ypqKuLy7NrIUha9WG159nhiuIyqYhW",
  "test": "1yKSi0vzJYAVmx_kRT7OFw4Ow86y_cKj0",
  "train": "1Q6imvE7v_KMlnxw_dQAMOVe5IuKUO6x3"
}


class ImagenetPatchCNNBias:
  def __init__(self, example_id, percents, poe):
    """
    :param example_id:
    :param percents: Percent of patch classifiers that predicted each class
    :param poe: Product-of-experiments ensemble prediction of all patch classifiers (not used
                currently since I thought `percents` did a bit job of capturing back-class
                correlations)
    """
    self.example_id = example_id
    self.percents = percents
    self.poe = poe


def load_imagenet_patch_bias(ds) -> List[ImagenetPatchCNNBias]:
  cache_name = join(IMAGENET_BIAS_CACHE, ds.split + ".tsv")
  if not exists(cache_name):
    download_from_drive(IMAGENET_BIAS_FILEIDS[ds.split], cache_name)
  out = []
  with open(cache_name) as f:
    f.readline()
    for line in f:
      parts = line.split()
      percent = np.array([float(x) for x in parts[1:7]])
      poe = np.array([float(x) for x in parts[7:]])
      out.append(ImagenetPatchCNNBias(parts[0], percent, poe))
  return out
