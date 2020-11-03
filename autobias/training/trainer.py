import json
import logging
import random
import socket
import sys
from collections import OrderedDict
from datetime import datetime
from os import makedirs
from os.path import join, exists
from typing import List

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  logging.warning("Unable to import SummaryWriter")
  SummaryWriter = None

import torch
from tqdm import trange, tqdm
from autobias.config import WEIGHTS_NAME, BEST_WEIGHTS_NAME
from autobias.datasets.dataset import Dataset
from autobias.training.data_iterator import DataIterator
from autobias.model.data_parallel import DataParallel
from autobias.training import train_utils
from autobias.training.evaluator import Evaluator
from autobias.training.optimizer import ConfiguredOptimizer
from autobias.utils import ops, py_utils
from autobias.utils.configured import Configured
import numpy as np


class PreEvalHook(Configured):
  """Hook that is run after training, but before evaluation"""

  def init(self, training_examples, collate_fn, split_batch=None):
    raise NotImplementedError()

  def post_fit(self, is_final, model, batch_to_device):
    raise NotImplementedError()


class EvalDataset(Configured):
  """Evaluation specification to use during training"""

  def __init__(self, dataset: Dataset, iterator: DataIterator, logging_name=None, evaluator=None):
    self.dataset = dataset
    self.iterator = iterator
    self.evaluator = evaluator
    self.logging_name = dataset.split if logging_name is None else logging_name


class StoppingPoint(Configured):
  def __init__(self, dataset_name, metric_name, tolerance_thresh, patience,
               min_steps=None):
    self.dataset_name = dataset_name
    self.metric_name = metric_name
    self.tolerance_thresh = tolerance_thresh
    self.patience = patience
    self.min_steps = min_steps
    self.cur_best = None
    self.n_fail = 0
    self.n_steps = 0

  def should_stop(self, results):
    self.n_steps += 1
    if self.min_steps and self.min_steps >= self.n_steps:
      return False

    score = results[self.dataset_name][self.metric_name]
    if self.cur_best is None:
      self.cur_best = score
      return False

    gain = self.cur_best - score

    if gain > 0:
      self.cur_best = score

    if gain > self.tolerance_thresh:
      self.n_fail = 0
      return False
    else:
      self.n_fail += 1
      if self.n_fail >= self.patience:
        logging.info(
          f"{self.n_fail} evals with improvement on {self.dataset_name}/{self.metric_name})")
        return True
      return False


class Trainer(Configured):

  def __init__(
      self,
      optimizer: ConfiguredOptimizer,
      train_dataset: Dataset,
      eval_sets: List[EvalDataset],
      num_train_epochs: int,
      train_iterator: DataIterator,
      train_eval_iterator: DataIterator=None,
      evaluator: Evaluator=None,
      pre_eval_hook: PreEvalHook=None,
      save_each_epoch=True,
      split_then_collate=True,
      save_best_model=None,
      early_stopping_criteria=None,

      # Logging
      eval_on_epochs=None,
      monitor_decay=None,
      tb_step_interval=10,
      tb_factor=1,
      dataset_logging_prefix="-",
      loss_logging_ema=0.998,
      log_to_tb=True,

      # Cosmetic
      evals_to_print=None,
      print_evals_as_table=True,
      progress_bar=True,
      eval_progress_bar=None,
      epoch_progress_bar=False,
      newline_before_results=False,
      print_eval_to=None
  ):
    """
    :param optimizer:
    :param train_dataset:
    :param eval_sets:
    :param num_train_epochs:
    :param train_iterator: Iterator to use to generate training batches
    :param train_eval_iterator: Iterator to use if evaluating on the training dataset
    :param evaluator: Evaluator to use when reporting evaluation results
    :param pre_eval_hook: Hook to run before evaluations
    :param save_each_epoch: Save the model each epoch
    :param split_then_collate: A somewhat experimental approach to multi-gpu training where
                               collate is called to build `n` batches, which are then sent
                               to individual GPUs instead collating once and then scatter the
                               resulting tensors. It was not really needed at the end of the day,
                               but the original models all ran with this on so I am going to
                               leave it on by default for consistency.
    :param save_best_model: Save the best model according to this given
                            (eval-dataset-name, metric-name) tuple
    :param early_stopping_criteria: Decides if we should stop early
    """
    if eval_sets or train_eval_iterator is not None:
      if evaluator is None:
        raise ValueError("Train evaluation requested, but no evaluator given!")
    self.early_stopping_criteria = early_stopping_criteria
    self.progress_bar = progress_bar
    self.eval_progress_bar = progress_bar if eval_progress_bar is None else eval_progress_bar
    self.split_then_collate = split_then_collate
    self.newline_before_results = newline_before_results
    self.save_best_model = save_best_model
    self.train_iterator = train_iterator
    self.tb_step_interval = tb_step_interval
    self.epoch_progress_bar = epoch_progress_bar
    self.evaluator = evaluator
    self.eval_on_epochs = eval_on_epochs
    self.train_dataset = train_dataset
    self.eval_sets = eval_sets
    self.train_eval_iterator = train_eval_iterator
    self.optimizer = optimizer
    self.loss_logging_ema = loss_logging_ema
    self.num_train_epochs = num_train_epochs
    self.log_to_tb = log_to_tb
    self.pre_eval_hook = pre_eval_hook
    self.monitor_decay = monitor_decay
    self.save_each_epoch = save_each_epoch
    self.tb_factor = tb_factor
    self.evals_to_print = evals_to_print
    self.print_evals_as_table = print_evals_as_table
    self.dataset_logging_prefix = dataset_logging_prefix
    self.print_eval_to = print_eval_to

  def train(self, model, output_dir, seed=None, *args, **kwargs):
    if output_dir:
      train_utils.clear_if_nonempty(output_dir)
      train_utils.init_model_dir(output_dir, self, model)
      if seed is not None:
        raise NotImplementedError()
      subdir = train_utils.select_subdir(output_dir)
      logging.info("Saving run to %s" % subdir)
    else:
      subdir = None
    self.training_run(model, subdir, seed, *args, **kwargs)

  def training_run(
      self, model, subdir, seed=None, n_processes=None, notes=None,
      fp16=False, no_cuda=False, print_out=sys.stdout):
    train_dataset = self.train_dataset
    eval_sets = self.eval_sets
    eval_sets = list(eval_sets)

    if self.evaluator is not None:
      self.evaluator.preprocess([train_dataset] + [x.dataset for x in eval_sets])

    if no_cuda is None or no_cuda is True:
      device = torch.device("cpu")
      n_gpu = 0
    else:
      device = torch.device("cuda")
      n_gpu = torch.cuda.device_count()

    logging.info("device: {} n_gpu: {}, 16-bits training: {}".format(device, n_gpu, fp16))

    if seed is not None:
      seed_init = np.random.RandomState(seed).randint(0, 2**32)
      random.seed(seed_init)
      np.random.seed(seed_init)
      torch.manual_seed(seed_init)
      if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_init)

    logging.info("Setting up data...")

    examples = model.preprocess_train(
      train_dataset, [x.dataset for x in eval_sets], n_processes)

    train_data = examples[0]

    split_batch = n_gpu > 1 and self.split_then_collate

    train_dataloader = self.train_iterator.build_iterator_for_model(
      train_data, model, n_gpu if split_batch else None)

    model_collate_fn = model.get_collate_fn()

    if split_batch:
      batch_to_device = lambda x: tuple(ops.to_device(part, dev) for part, dev in zip(x, model.device_ids))

      def with_batch_collate_fn(collate_examples):
        return tuple((x, model_collate_fn(x)) for x in py_utils.split(collate_examples, n_gpu))

    else:
      batch_to_device = lambda _x: ops.to_device(_x, device)

      def with_batch_collate_fn(collate_examples):
        return collate_examples, model_collate_fn(collate_examples)

    if self.pre_eval_hook:
      self.pre_eval_hook.init(train_data, model_collate_fn, n_gpu if split_batch else None)

    if self.evaluator:
      eval_loaders = OrderedDict()
      for ds, ex in zip(eval_sets, examples[1:]):
        loader = ds.iterator.build_iterator(ex, with_batch_collate_fn)
        evaluator = ds.evaluator if ds.evaluator is not None else self.evaluator
        eval_loaders[ds.logging_name] = (loader, evaluator)
      if self.train_eval_iterator:
        train_eval_loader = self.train_eval_iterator.build_iterator(train_data, with_batch_collate_fn)
        eval_loaders["train"] = (train_eval_loader, self.evaluator)
    else:
      eval_loaders = None

    num_train_optimization_steps = len(train_dataloader) * self.num_train_epochs

    logging.info("Initializing...")
    model.reset_parameters()  # Run this after setting the seed so intialization is the same

    logging.info("Setting up model...")
    model.to(device)

    # Prepare optimizer
    # Do this last (after model.to and mode.preprocess_train) since those steps
    # might add parameters
    opt_model = self.optimizer.set_model(model, num_train_optimization_steps, fp16)
    if opt_model is not None:
      model = opt_model

    if n_gpu > 1:
      single_gpu_model = model
      model = DataParallel(model, self.split_then_collate)
    else:
      single_gpu_model = model

    if subdir:
      if not exists(subdir):
        makedirs(subdir, exist_ok=True)
        logging.info("Saving run to %s" % subdir)

      model_output = subdir

      if notes is not None:
        with open(join(model_output, "notes.txt"), "w") as f:
          f.write(notes)

      with open(join(model_output, "runtime.json"), "w") as f:
        json.dump(dict(
          hostname=socket.gethostname(), date=datetime.now().strftime("%m%d-%H%M%S"),
          n_processes=n_processes, n_gpu=n_gpu, fp16=fp16, seed=seed
        ), f, indent=2)

        output_model_file = join(model_output, WEIGHTS_NAME)
        status_file = join(model_output, "status.json")
    else:
      status_file = None
      output_model_file = None
      model_output = None

    if self.log_to_tb and model_output:
      summary_writer = SummaryWriter(join(model_output, "train-log"))
    else:
      summary_writer = None

    best_model_score = None

    def run_evaluations(_global_step):
      if not self.evaluator:
        return
      if self.pre_eval_hook:
        self.pre_eval_hook.post_fit(False, model, batch_to_device)
      model.eval()

      eval_results = OrderedDict()
      for name, (loader, evaluator) in eval_loaders.items():
        evaluator.clear()
        if n_gpu > 1 and self.split_then_collate:
          for group in tqdm(loader, desc="eval-%s" % name, ncols=100, disable=not self.eval_progress_bar):
            examples, batches = py_utils.transpose_lists(group)
            batches = batch_to_device(batches)
            with torch.no_grad():
              outputs = model.forward(*batches)
              for lst, out, (_, label) in zip(examples, outputs, batches):
                evaluator.evaluate_batch(lst, out, label)
        else:
          for example_lst, batch in tqdm(loader, desc="eval-%s" % name, ncols=100, disable=not self.eval_progress_bar):
            batch = batch_to_device(batch)
            with torch.no_grad():
              out = model.forward(*batch)
              if n_gpu > 1:
                # This is a bit of hack since it trusts there is a `logprobs` attribute
                lens = [x.logprobs.size(0) for x in out]
                example_lst = py_utils.split_list(example_lst, lens)
                label_chunks = py_utils.split_list(batch[-1], lens)
                for lst, chunk_out, label_chunk in zip(example_lst, out, label_chunks):
                  label_chunk = label_chunk.to(chunk_out.logprobs.device)
                  evaluator.evaluate_batch(lst, chunk_out, label_chunk)
              else:
                evaluator.evaluate_batch(example_lst, out, batch[-1])
        results = evaluator.get_stats()
        eval_results[name] = results

        if summary_writer is not None:
          for k, v in results.items():
            summary_writer.add_scalar(
              name + self.dataset_logging_prefix + k, v, int(_global_step * self.tb_factor))

      if self.newline_before_results:
        print_out.write("\n")

      if self.save_best_model and model_output is not None:
        cur_score = eval_results[self.save_best_model[0]][self.save_best_model[1]]
        nonlocal best_model_score
        if best_model_score is None or cur_score > best_model_score:
          if best_model_score is None:
            logging.info(f"Saving best-model ({cur_score:.4f} {self.save_best_model[1]})")
          else:
            logging.info(f"Saving best-model ({cur_score:.4f} > {best_model_score:.4f} {self.save_best_model[1]})")
          best_model_score = cur_score
          torch.save(single_gpu_model.get_state(), join(model_output, BEST_WEIGHTS_NAME))

      if self.print_evals_as_table:
        if self.evals_to_print is not None:
          keys = self.evals_to_print
        else:
          keys = set()
          ordered_keys = list()
          for r in eval_results.values():
            for key in r:
              if key not in keys:
                keys.add(key)
                ordered_keys.append(key)
          keys = ordered_keys

        rows = [["name"] + keys]
        for name, r in eval_results.items():
          rows.append([name] + [("%.4f" % r[k] if k in r else "-") for k in keys])
        py_utils.print_table(rows, out=print_out)
        print_out.flush()
      else:
        for name, r in eval_results.items():
          print(name + ": " + " ".join("%s=%.4f" % (k, v) for k, v in r.items()))
        sys.stdout.flush()
      model.train()
      return eval_results

    loss_ema = 0
    n_steps = 0
    global_step = 0
    loss_decay = self.loss_logging_ema
    descript = "loss=%.4f" % 0.0
    model.train()

    if self.epoch_progress_bar:
      it = trange(1, int(self.num_train_epochs)+1, desc="epoch", ncols=100)
    else:
      it = range(1, int(self.num_train_epochs)+1)

    monitor_ema = {}
    if self.monitor_decay:
      for k in self.monitor_decay:
        monitor_ema[k] = 0

    stop_early = False
    for epoch_num in it:
      if stop_early:
        break

      pbar = tqdm(train_dataloader, desc=descript, ncols=100, disable=not self.progress_bar)

      for i, batch in enumerate(pbar):
        batch = batch_to_device(batch)
        out = model(*batch)

        loss = out.loss
        monitor = out.monitor

        batch_loss = loss.item()
        if not np.isfinite(batch_loss):
          raise ValueError("non-finite loss %s" % batch_loss)

        self.optimizer.backwards(loss)

        loss_ema = loss_ema * loss_decay + batch_loss * (1 - loss_decay)
        n_steps += 1
        corrected_loss_ema = (loss_ema / (1 - loss_decay**n_steps))
        if self.evaluator:
          descript = "ep%d loss=%.4f" % (epoch_num, corrected_loss_ema)
        else:
          descript = "loss=%.4f" % corrected_loss_ema

        pbar.set_description(descript, refresh=False)

        if monitor is not None and monitor_ema is not None:
          for k, v in monitor.items():
            if k in monitor_ema:
              v = ops.numpy(v)
              decay = self.monitor_decay[k]
              v = monitor_ema[k] * decay + v * (1 - decay)
              monitor_ema[k] = v
              monitor[k] = (v / (1 - decay ** n_steps))

        if monitor is None:
          monitor = dict(loss=corrected_loss_ema)
        elif "loss" not in monitor:
          monitor["loss"] = corrected_loss_ema

        if summary_writer is not None and (global_step + 1) % self.tb_step_interval == 0:
          summary_writer.add_scalar(
            "loss", batch_loss, int(global_step * self.tb_factor))
          if monitor is not None:
            for k, v in monitor.items():
              summary_writer.add_scalar(
                "monitor/" + k, v, int(global_step * self.tb_factor))

        self.optimizer.step(epoch_num)
        global_step += 1

        # End epoch loop

      if model_output is not None and self.save_each_epoch is not None:
        if isinstance(self.save_each_epoch, bool):
          save = self.save_each_epoch
        else:
          save = epoch_num % self.save_each_epoch == 1
        if save:
          logging.info("Saving model")
          torch.save(single_gpu_model.get_state(), output_model_file)
          py_utils.write_json(dict(step=global_step, epoch=epoch_num, done=False), status_file, 2)

      if self.eval_on_epochs is None:
        do_eval = True
      elif isinstance(self.eval_on_epochs, bool):
        do_eval = self.eval_on_epochs
      elif isinstance(self.eval_on_epochs, int):
        do_eval = epoch_num % self.eval_on_epochs == 1
      elif self.eval_on_epochs is not None:
        do_eval = epoch_num in self.eval_on_epochs
      else:
        raise ValueError()

      if do_eval:
        eval_results = run_evaluations(global_step)
        if self.early_stopping_criteria and self.early_stopping_criteria.should_stop(eval_results):
          stop_early = eval_results
          break

    if self.pre_eval_hook:
      self.pre_eval_hook.post_fit(False, model, batch_to_device)

    if not model_output:
      # TODO support running the evalation even without an otuput dir
      return

    if summary_writer is not None:
      summary_writer.close()

    # Save a trained model and the associated configuration
    logging.info("Saving model")
    torch.save(single_gpu_model.get_state(), output_model_file)
    py_utils.write_json(dict(step=global_step, epoch=epoch_num, done=True), status_file, 2)

    return stop_early

