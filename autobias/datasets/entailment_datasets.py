import logging
from os import listdir
from os.path import join, exists
from typing import List, Dict

import numpy as np

from autobias import config
from autobias.datasets.dataset import Dataset
from autobias.utils import downloader, py_utils

HANS_URL = "https://raw.githubusercontent.com/tommccoy1/hans/master/heuristics_evaluation_set.txt"
# Taken from the GLUE script
MNLI_URL = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce"


NLI_LABELS = ["contradiction", "entailment", "neutral"]
NLI_LABEL_MAP = {k: i for i, k in enumerate(NLI_LABELS)}


class TextPairExample:
  """Example for text pair classification"""

  def __init__(self, example_id: str, text_a: str, text_b: str, label, other_features=None):
    self.example_id = example_id
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.other_features = other_features

  def __str__(self):
    if self.text_b is not None:
      return "\"%s\" => \"%s\" (label=%s)" % (self.text_a, self.text_b, self.label)
    else:
      return "\"%s\" (label=%s)" % (self.text_a, self.label)


class EntailmentDataset(Dataset):
  TWO_CLASS_NAMES = ["non-entailment", "entailment"]
  THREE_CLASS_NAMES = ["contradiction", "entailment", "neutral"]

  def __init__(self, domain, split, three_class, sample_name=None):
    super().__init__(domain, split, sample_name)
    self.three_class = three_class

  def _load(self) -> List[TextPairExample]:
    raise NotImplementedError()

  def n_classes(self):
    return 3 if self.three_class else 2


class _MnliBase(Dataset):
  def __init__(self, split, src, example_id_prefix, sample=None):
    self.src = src
    self.example_id_prefix = example_id_prefix
    self.sample = sample
    sample_name = None
    if sample:
      if sample % 1000 == 0:
        sample_name = str(sample // 1000) + "k"
      else:
        sample_name = str(sample)
    super().__init__("mnli", split, sample_name)

  def _load(self):
    mnli_source = join(config.GLUE_DATA, "MNLI")
    if not (exists(mnli_source) and len(listdir(mnli_source)) > 0):
      downloader.download_zip("MNLI", MNLI_URL, config.GLUE_DATA)

    filename = join(config.GLUE_DATA, "MNLI", self.src)
    logging.info("Loading %s", filename)

    with open(filename) as f:
      f.readline()
      lines = f.readlines()

    if self.sample:
      if len(lines) < self.sample:
        raise ValueError("Requested a sample of %d, but only %d items" % (self.sample, len(lines)))
      np.random.RandomState(26096781 + self.sample).shuffle(lines)
      lines = lines[:self.sample]

    out = []
    for i, line in enumerate(lines):
      line = line.strip().split("\t")
      ex_id = line[0]
      if self.example_id_prefix:
        ex_id = self.example_id_prefix + ex_id
      out.append(TextPairExample(ex_id, line[8], line[9], NLI_LABEL_MAP[line[-1]]))
    return out


class MnliTrain(_MnliBase):
  def __init__(self, sample=None):
    super().__init__("train", "train.tsv", "tr", sample)


class MnliDevMatched(_MnliBase):
  def __init__(self, sample=None):
    super().__init__("dev_matched", "dev_matched.tsv", "dev", sample)


class MnliDevUnmatched(_MnliBase):
  def __init__(self, sample=None):
    super().__init__("dev_unmatched", "dev_mismatched.tsv", "dev", sample)


class Hans(Dataset):
  def __init__(self, sample=None):
    super().__init__("HANS", "test", None if sample is None else str(sample))
    self.n_samples = sample

  def _load(self):
    logging.info("Loading hans...")
    src = config.HANS
    if not exists(src):
      logging.info("Downloading source to %s..." % config.HANS)
      downloader.download_to_file(HANS_URL, src)

    out = []
    with open(src, "r") as f:
      f.readline()
      lines = f.readlines()

    n_samples = self.n_samples
    if n_samples is not None:
      lines = np.random.RandomState(16349 + n_samples).choice(lines, n_samples, replace=False)

    for line in lines:
      parts = line.split("\t")
      label = parts[0]
      if label == "non-entailment":
        label = 0
      elif label == "entailment":
        label = 1
      else:
        raise RuntimeError()
      s1, s2, pair_id = parts[5:8]
      out.append(TextPairExample(pair_id, s1, s2, label))
    return out


HYPOTHESIS_BIAS = {
  "mnli-dev_unmatched": "1xIP9Bg1YOvCWbbziKPzcs52jimQgMEF0",
  "mnli-train": "1HVOLWVR8sug5iEGNc6aOaFo4kQjgFVuD",
  "mnli-dev_matched": "1MxCtoK2iE4F5lazYT1D7ZhNn6rijHpKw",

  # For debug runs
  "mnli-dev_matched-512": "1IsyKnwM_OuJ-Z1O-l41xLe8SDAXZ9yJT",
  "mnli-train-4096": "1rnpIZq-fDL35tZKiTH-JSh3Xm7RUEKgw",
}


def load_hypothesis_bias(dataset_name) -> Dict[str, np.ndarray]:
  """Load dictionary of example_id->bias where bias is a length 3 array
  of log-probabilities"""

  if dataset_name not in HYPOTHESIS_BIAS:
    raise ValueError(dataset_name)
  cache = config.MNLI_BIAS_CACHE
  cache_file = join(cache, dataset_name + "-predictions.pkl")
  if not exists(cache_file):
    logging.info("Downloading MNLI bias to %s..." % cache_file)
    downloader.download_from_drive(HYPOTHESIS_BIAS[dataset_name], cache_file)

  return py_utils.load_pickle(cache_file)
