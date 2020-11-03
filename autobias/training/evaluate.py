import logging
import pickle
from collections import defaultdict
from os import mkdir
from os.path import join, exists

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from autobias.model.data_parallel import DataParallel
from autobias.utils import ops, py_utils


def run_on_evaluator(examples, model, batch_size, evaluator, no_cuda=False,
                     progress_bar=True, n_workers=None):
  """Get the evaluation results for the given list of examples"""

  device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")

  model.to(device)

  n_gpu = torch.cuda.device_count()
  if n_gpu > 1:
    fn = model.get_collate_fn()

    def collate_fn(batch):
      return tuple((x, fn(x)) for x in py_utils.split(batch, n_gpu))

    model = DataParallel(model, True)
  else:
    fn = model.get_collate_fn()
    collate_fn = lambda x: (x, fn(x))

  evaluator.clear()
  model.eval()

  if batch_size is None:
    if n_gpu > 1:
      raise NotImplementedError()
    examples, batch = collate_fn(examples)
    batch = ops.to_device(batch, device)
    out = model.forward(*batch)
    evaluator.evaluate_batch(examples, out, batch[-1])
    return evaluator.get_stats()

  dataloader = DataLoader(
    examples,
    sampler=SequentialSampler(examples),
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=0 if n_workers is None else n_workers
  )

  for batch in tqdm(dataloader, desc="eval", ncols=100, disable=not progress_bar):
    with torch.no_grad():
      if n_gpu > 1:
        examples = [part[0] for part in batch]
        batch = [ops.to_device(part[1], dev) for part, dev in zip(batch, model.device_ids)]
        for sub_ex, sub_out, sub_batch in zip(examples, model.forward(*batch), batch):
          evaluator.evaluate_batch(sub_ex, sub_out, sub_batch[-1])
      else:
        examples, batch = batch
        batch = ops.to_device(batch, device)
        out = model.forward(*batch)
        evaluator.evaluate_batch(examples, out, batch[-1])

  return evaluator.get_stats()


def get_cached_evaluations(models, dataset, evaluator_name):
  """Get all cached evaluations"""
  all_stats = defaultdict(list)
  not_cached_models = {}
  for name, (model_dir, runs) in models.items():
    not_cached = []
    for run in runs:
      result = get_cached_evaluation(run, dataset, evaluator_name)
      if result is None:
        not_cached.append(run)
      else:
        all_stats[name].append(result)
    if len(not_cached) > 0:
      not_cached_models[name] = (model_dir, not_cached)
  models = not_cached_models
  return models, all_stats


def get_cached_evaluation(run, dataset, evaluator_name):
  """Get cached evaluations if one exists"""
  if evaluator_name is None:
    return None
  eval_result_file = join(run, "eval", dataset.fullname + "-" + evaluator_name + ".json")
  if exists(eval_result_file):
    return py_utils.load_json(eval_result_file)


def run_evaluation(run, dataset, batch_size, evaluator, evaluator_name=None,
                   cache=False, sort=True,
                   n_processes=None, best=False, model=None,
                   progress_bar=True, n_workers=None):
  """Get the results of running `evaluator` on `dataset` using the model saved in `run`.

  Supports caching using `evaluator_name` and the dataset name as a key
  """
  eval_cache = join(run, "eval")
  if cache and not exists(eval_cache):
    mkdir(eval_cache)

  if cache and evaluator_name is None:
    raise ValueError("Unable to use cache without an evaluation name")

  if evaluator_name is not None:
    eval_result_file = join(eval_cache, dataset.fullname + "-" + evaluator_name + ".json")
  else:
    eval_result_file = None

  if cache and exists(eval_result_file):
    logging.info(f"Loading results from {eval_result_file}")
    stats = py_utils.load_json(eval_result_file)
  else:
    if model is None:
      from autobias.utils.load_config import load_model
      model = load_model(run, best=best)

    examples = model.preprocess_dataset(dataset, n_processes)
    if sort:
      examples.sort(key=lambda x: x.get_len())
    stats = run_on_evaluator(examples, model, batch_size, evaluator,
                             progress_bar=progress_bar, n_workers=n_workers)
    if cache:
      logging.info(f"Saving results to {eval_result_file}")
      py_utils.write_json(stats, eval_result_file, indent=2)
  return stats


def get_evaluation(models, nocache, dataset, evaluator, evaluator_name, batch_size,
                   n_workers=None, sort=True, progress_bar=True, n_processes=4):
  """Run an evaluation over a set of models."""
  if not nocache:
    models, all_stats = get_cached_evaluations(models, dataset, evaluator_name)
    cached = sum(len(run_stats) for run_stats in all_stats.values())
    to_run = sum(len(runs) for _, runs in models.values())
    if to_run == 0:
      logging.info("All models already have cached results")
    else:
      logging.info(f"{cached} models cached, {to_run} to complete")
  else:
    all_stats = defaultdict(list)

  dataset.cache = True

  total_runs = sum(len(x[1]) for x in models.values())
  total_evaluated = 0
  for name, (model_dir, runs) in models.items():
    logging.info("\n")
    logging.info(f"Evaluating model: {name} ({len(runs)} runs) ({total_evaluated+1}/{total_runs})")
    for run in runs:
      stats = run_evaluation(
        run, dataset, batch_size, evaluator, evaluator_name,
        n_workers=n_workers, sort=sort, cache=not nocache,
        n_processes=n_processes, progress_bar=progress_bar)
      all_stats[name].append(stats)
      total_evaluated += 1
  return all_stats
