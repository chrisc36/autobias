import logging
import re
from bisect import bisect_right
from typing import List

from torch import nn, optim

from autobias.utils import py_utils
from autobias.utils.configured import Configured


class ConfiguredOptimizer(Configured):
  """Configured class that does optimization"""
  def set_model(self, model: nn.Module, num_train_steps, fp16):
    raise NotImplementedError()

  def backwards(self, loss):
    raise NotImplementedError()

  def step(self, epoch):
    raise NotImplementedError()


class LearningRateShedule(Configured):
  def get_lr(self, step, t_total, epoch):
    raise NotImplementedError()


class PiecewiseLinear(Configured):
  def __init__(self, epochs, gamma):
    self.epochs = epochs
    self.gamma = gamma

  def get_lr(self, step, _, epoch):
    return self.gamma**bisect_right(self.epochs, epoch)


class LinearWarmup(Configured):
  def __init__(self, warmup_steps):
    self.warmup_steps = warmup_steps

  def get_lr(self, step, t_total, epoch):
    if step > self.warmup_steps:
      return 1
    else:
      return step / self.warmup_steps


class LinearTriangle(Configured):
  def __init__(self, warmup_steps):
    self.warmup_steps = warmup_steps

  def get_lr(self, step, t_total, epoch):
    x = step / t_total
    if x < self.warmup_steps:
      return x/self.warmup_steps
    return 1.0 - x


class ConstantLearningRate(Configured):
  def get_lr(self, step, t_total, epoch):
    return 1.0


class ParameterSet(Configured):
  def __init__(self, set_name, regex, opt_parameters, schedule=None, fp16=True):
    self.fp16 = fp16
    self.set_name = set_name
    self.schedule = schedule
    self.regex = regex
    self.opt_parameters = opt_parameters


class Adam(ConfiguredOptimizer):

  def __init__(
      self, lr=0.001, b1=0.9, b2=0.999, e=1e-8, weight_decay=0.0,
      max_grad_norm=3.0, loss_scale=None, schedule=None,
      alternative_sets: List[ParameterSet]=None
  ):
    self.lr = lr
    self.loss_scale = loss_scale
    self.b1 = b1
    self.b2 = b2
    self.e = e
    self.weight_decay = weight_decay
    self.max_grad_norm = max_grad_norm
    self.loss_scale = loss_scale
    self.schedule = schedule
    self.alternative_sets = alternative_sets
    self._lrs = None

    self.on_step = 0
    self.num_train_steps = None
    self._opt = None
    self.fp16 = None

  def backwards(self, loss):
    if self.fp16:
      self._opt.backward(loss)
    else:
      loss.backward()

  def step(self, epoch):
    if self.max_grad_norm and not self.fp16:
        params = py_utils.flatten_list(group['params'] for group in self._opt.param_groups)
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
    if self.schedule:
      self.on_step += 1
      decay = self.schedule.get_lr(self.on_step, self.num_train_steps, epoch)
      if self.alternative_sets is not None:
        for param_group, lr, opt in zip(self._opt.param_groups, self._lrs, self.alternative_sets):
          if lr is None:
            continue
          if opt.schedule is not None:
            d = opt.schedule.get_lr(self.on_step, self.num_train_steps, epoch)
          else:
            d = decay
          param_group['lr'] = lr * d
      else:
        for param_group in self._opt.param_groups:
          param_group["lr"] = self.lr * decay

    self._opt.step()
    self._opt.zero_grad()

  def set_model(self, model: nn.Module, num_train_steps, fp16):
    self.num_train_steps = num_train_steps

    if fp16:
      model = model.half()

    if self.alternative_sets:
      param_optimizer = list(model.named_parameters())
      total = len(param_optimizer)
      params = []
      self._lrs = []
      for alt_set in self.alternative_sets:
        prefix = re.compile(alt_set.regex)
        other_params = []
        alter_params = []
        for n, p in param_optimizer:
          if prefix.match(n):
            alter_params.append(p)
          else:
            other_params.append((n, p))
        if len(alter_params) == 0:
          self._lrs.append(None)
          logging.warning(f"No parameters for group {alt_set.set_name}")
          continue
        logging.info(f"{len(alter_params)}/{total} parameters in set {alt_set.set_name}")
        opt_parameters = dict(alt_set.opt_parameters)
        opt_parameters["params"] = alter_params
        params.append(opt_parameters)
        param_optimizer = other_params
        self._lrs.append(alt_set.opt_parameters.get("lr", self.lr))

      if len(param_optimizer) > 0:
        logging.info(f"{len(param_optimizer)}/{total} remaining parameters")
        params.append({'params': [x[1] for x in param_optimizer]})
    else:
      params = list(model.parameters())
      self._lrs = self.lr

    self.fp16 = fp16
    from apex.optimizers import FusedAdam
    self._opt = FusedAdam(
      params, lr=self.lr, betas=(self.b1, self.b2),
      eps=self.e, weight_decay=self.weight_decay,
      bias_correction=True,
      # If not fp16, we clip the grad norm ourself
      max_grad_norm=self.max_grad_norm if fp16 else 0
    )

    if fp16:
      from apex.optimizers import FP16_Optimizer
      if self.loss_scale is None:
        self._opt = FP16_Optimizer(self._opt, dynamic_loss_scale=True, verbose=False)
      else:
        self._opt = FP16_Optimizer(self._opt, static_loss_scale=self.loss_scale, verbose=False)
      return model


class SGD(ConfiguredOptimizer):
  version = 2

  def __init__(self, lr, momentum, weight_decay=0, clip_grad_norm=None, schedule=None):
    self.lr = lr
    self.clip_grad_norm = clip_grad_norm
    self.momentum = momentum
    self.weight_decay = weight_decay
    self.schedule = schedule
    self.optim = None
    self.num_train_steps = None
    self.on_step = 0

  def set_model(self, model: nn.Module, num_train_steps, fp16):
    if fp16:
      raise NotImplementedError()
    self.optim = optim.SGD(
      model.parameters(), self.lr, self.momentum, weight_decay=self.weight_decay)
    self.num_train_steps = num_train_steps

  def backwards(self, loss):
    loss.backward()

  def step(self, epoch):
    if self.clip_grad_norm:
      for group in self.optim.param_groups:
        nn.utils.clip_grad_norm_(group['params'], self.clip_grad_norm)
    if self.schedule:
      self.on_step += 1
      decay = self.schedule.get_lr(self.on_step, self.num_train_steps, epoch)
      for param_group in self.optim.param_groups:
        param_group["lr"] = self.lr * decay

    self.optim.step()
    self.optim.zero_grad()

