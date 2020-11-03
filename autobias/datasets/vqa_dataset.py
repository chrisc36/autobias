import logging
import pickle
import sys
from os import makedirs
from os.path import join, exists
from typing import List, Tuple

import numpy as np

from autobias import config
from autobias.datasets.dataset import Dataset
from autobias.utils import py_utils


class VqaExample:
  def __init__(self, example_id: str, image_id: str, question: str,
               question_type: str, answer_type, multiple_choice_answer: str, answers: Tuple[str]):
    self.example_id = example_id
    self.image_id = image_id
    self.question = question
    self.answers = answers
    self.question_type = question_type
    self.answer_type = answer_type
    self.multiple_choice_answer = multiple_choice_answer


class VqaBase(Dataset):
  def __init__(self, sample, domain, split, q_file, anno_file):
    if sample is None:
      sample_name = None
    else:
      sample_name = str(sample)
    self.domain = domain
    self.sample = sample
    self.q_file = q_file
    self.anno_file = anno_file
    super().__init__(domain, split, sample_name)

  def _load(self) -> List[VqaExample]:
    cache_name = join(config.VQA_CACHE, self.fullname + ".pkl")
    if exists(cache_name):
      logging.info(f"Loading {self.fullname} from cache")
      return py_utils.load_pickle(cache_name)
    else:
      logging.info(f"Loading {self.fullname} from json")
      makedirs(config.VQA_CACHE, exist_ok=True)
      anno = py_utils.load_json(self.anno_file)
      if "annotations" in anno:
        anno = anno["annotations"]
      questions = py_utils.load_json(self.q_file)
      if "questions" in questions:
        questions = questions["questions"]

      if len(anno) != len(questions):
        raise RuntimeError()
      anno = {x["question_id"]: x for x in anno}
      out = []
      for q in questions:
        a = anno[q["question_id"]]
        # Interning reduces RAM, and making loading from pickle a bit faster
        out.append(VqaExample(
          q["question_id"], q["image_id"], q["question"],
          sys.intern(a["question_type"]), sys.intern(a["answer_type"]),
          sys.intern(a["multiple_choice_answer"]), tuple(sys.intern(x["answer"]) for x in a["answers"])
        ))

      if self.sample:
        np.random.RandomState(5819 + 40000).shuffle(out)
        out = out[:self.sample]

      logging.info(f"Caching...")
      with open(cache_name, "wb") as f:
        pickle.dump(out, f)

    return out


class Vqa2Train(VqaBase):
  def __init__(self, sample=None):
    super().__init__(
      sample, "vqa-2.0", "train",
      join(config.VQA_2, "v2_OpenEnded_mscoco_train2014_questions.json"),
      join(config.VQA_2, "v2_mscoco_train2014_annotations.json"),
    )


class Vqa2Dev(VqaBase):
  def __init__(self, sample=None):
    super().__init__(
      sample, "vqa-2.0", "dev",
      join(config.VQA_2, "v2_OpenEnded_mscoco_val2014_questions.json"),
      join(config.VQA_2, "v2_mscoco_val2014_annotations.json")
    )


class VqaCP2Train(VqaBase):
  def __init__(self, sample=None):
    super().__init__(
      sample, "vqacp-2.0", "train",
      join(config.VQA_CP, "vqacp_v2_train_questions.json"),
      join(config.VQA_CP, "vqacp_v2_train_annotations.json"),
    )


class VqaCP2Test(VqaBase):
  def __init__(self, sample=None):
    super().__init__(
      sample, "vqacp-2.0", "test",
      join(config.VQA_CP, "vqacp_v2_test_questions.json"),
      join(config.VQA_CP, "vqacp_v2_test_annotations.json"),
    )
