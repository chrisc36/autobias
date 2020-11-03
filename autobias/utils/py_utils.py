import json
import logging
import pickle
import sys
from os import listdir, walk
from os.path import join, exists, isdir, relpath, basename
from typing import List, TypeVar, Iterable

import numpy as np

from autobias.config import WEIGHTS_NAME

T = TypeVar('T')


def flatten_list(iterable_of_lists: Iterable[Iterable[T]]) -> List[T]:
  """Unpack lists into a single list."""
  return [x for sublist in iterable_of_lists for x in sublist]


def group(lst: List[T], max_group_size) -> List[List[T]]:
  """ partition `lst` into that the mininal number of groups that as evenly sized
  as possible  and are at most `max_group_size` in size """
  if max_group_size is None:
    return [lst]
  if max_group_size == 1:
    return [[x] for x in lst]
  n_groups = (len(lst) + max_group_size - 1) // max_group_size
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def split(lst: List[T], n_groups) -> List[List[T]]:
  """ partition `lst` into `n_groups` that are as evenly sized as possible  """
  per_group = len(lst) // n_groups
  remainder = len(lst) % n_groups
  groups = []
  ix = 0
  for _ in range(n_groups):
    group_size = per_group
    if remainder > 0:
      remainder -= 1
      group_size += 1
    groups.append(lst[ix:ix + group_size])
    ix += group_size
  return groups


def load_json(x):
  with open(x) as f:
    return json.load(f)


def write_json(obj, filename, indent=None):
  with open(filename, "w") as f:
    json.dump(obj, f, indent=indent)


class BackwardsCompatUnpickler(pickle.Unpickler):

  def find_class(self, module, name):
    if module == "autobias.model.text_pair_predictors":
      module = "autobias.modules.classifier_ensembles"
    return super().find_class(module, name)


def load_pickle(x):
  with open(x, "rb") as f:
    return BackwardsCompatUnpickler(f).load()


def write_pickle(obj, filename):
  with open(filename, "wb") as f:
    pickle.dump(obj, f)


def save_divide_np(a, b) -> np.ndarray:
  return np.divide(a, b, out=np.zeros_like(a), where=b!=0)


def print_table(table, out=sys.stdout):
  """Print the list of strings with evenly spaced columns."""
  # print while padding each column to the max column length
  col_lens = [0] * len(table[0])
  for row in table:
    for i, cell in enumerate(row):
      col_lens[i] = max(len(cell), col_lens[i])

  formats = ["{0:<%d}" % x for x in col_lens]
  for row in table:
    out.write(" ".join(formats[i].format(row[i]) for i in range(len(row))))
    out.write("\n")


def transpose_lists(lsts):
  """Transpose a list of lists."""
  return [list(i) for i in zip(*lsts)]


def add_stdout_logger():
  handler = logging.StreamHandler(sys.stdout)
  formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%m/%d %H:%M:%S', )
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)

  root = logging.getLogger()
  root.handlers = []  # Remove stderror handler that sometimes appears by default
  root.setLevel(logging.INFO)
  root.addHandler(handler)
  # transformers is pretty noisy, tell it to calm down
  logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)


def print_starred(text, n=10):
  print("*" * n + " " + text + " " + "*"*n)


def split_list(lst, lens):
  if len(lens) == 1:
    return [lst]
  on = 0
  out = []
  for l in lens:
    out.append(lst[on:on+l])
    on += l
  return out


def is_model_dir(x):
  return exists(join(x, "model.json"))


def is_run_dir(x):
  if not isdir(x):
    return False
  if exists(join(x, "status.json")):
    return load_json(join(x, "status.json"))["done"]
  return exists(join(x, WEIGHTS_NAME))


def extract_runs(model_dir):
  runs = []
  for run_dir in listdir(model_dir):
    run_dir = join(model_dir, run_dir)
    if is_run_dir(run_dir):
      runs.append(run_dir)
  return runs


def extract_model_group(src):
  out = {}
  if is_model_dir(src):
    runs = extract_runs(src)
    if len(runs) > 0:
      out[""] = (src, extract_runs(src))
  else:
    for subdir_name in listdir(src):
      subdir = join(src, subdir_name)
      if isdir(subdir):
        runs = extract_runs(subdir)
        if len(runs) > 0:
          out[subdir_name] = (subdir, extract_runs(subdir))
  return out


def extract_models(roots, require_runs=True):
  """Extract all models found in `roots`

  :param roots: Directories to recursively check for models
  :param require_runs: Only include models with at least one run
  :return: Dictionary of (model_name -> (model_dir, list of individual runs directories))
  """
  if isinstance(roots, str):
    roots = [(None, roots)]
  elif isinstance(roots, dict):
    roots = list(roots.items())
  elif len(roots) == 1:
    roots = [(None, roots[0])]
  else:
    split_roots = [x.split("/") for x in roots]
    prefix_len = 1
    while len(set(tuple(x[-prefix_len:]) for x in split_roots)) != len(roots):
      prefix_len += 1
    root_names = [join(*x[-prefix_len:]) for x in split_roots]
    roots = list(zip(root_names, roots))

  models = {}
  for root_name, root in roots:
    if is_model_dir(root):
      runs = []
      for run_dir in listdir(root):
        run_dir = join(root, run_dir)
        if is_run_dir(run_dir):
          runs.append(run_dir)
      model_name = basename(root)
      if root_name:
        model_name = join(root_name, model_name)
      models[model_name] = (root, runs)
      continue

    for dirpath, dirnames, filenames in walk(root):
      for model_dir in dirnames:
        model_dir = join(dirpath, model_dir)
        if not is_model_dir(model_dir):
          continue

        model_name = relpath(model_dir, root)
        if root_name:
          model_name = join(root_name, model_name)

        runs = extract_runs(model_dir)
        if not require_runs or len(runs) > 0:
          models[model_name] = (model_dir, runs)

  return models
