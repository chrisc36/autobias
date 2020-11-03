from collections import OrderedDict
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from autobias.argmin_modules.argmin_transform import ArgminTransformFunction
from autobias.model.model import TrainOutput, Predictor
from autobias.modules.layers import Layer
from autobias.utils import ops


class ClassifierOutput:
  def __init__(self, logprobs):
    self.logprobs = logprobs


class ClassifierEnsembleOutput:
  """Prediction made by an ensemble of models"""

  def __init__(self, logprobs, head_names, prior):
    """
    :param logprobs: [batch, n_classes, n_heads]  prediction per example, per head
    :param head_names: [n_heads], string name of each head
    :param prior: optional prior that should be integrated into the final predictions
    """
    self.logprobs = logprobs
    self.base = prior
    self.head_names = head_names


class ClfHead(Layer):
  """Head to use as part of an ensemble"""

  @classmethod
  def build(cls, version, params):
    # Backwards compatibility
    if "rescaler_builder" in params:
      params["rescaler"] = params.pop("rescaler_builder")
    return super().build(version, params)

  def __init__(self, predictor: Predictor, rescaler: ArgminTransformFunction=None,
               head_name=None, nll_penalty=None):

    super().__init__()
    if predictor is None:
      raise ValueError()
    self.predictor = predictor
    self.rescaler = rescaler
    self.head_name = head_name
    self.nll_penalty = nll_penalty

  def reset_parameters(self):
    pass

  def get_state(self):
    if self.rescaler is not None:
      return self.predictor.get_state(), self.rescaler.state_dict()
    else:
      return self.predictor.get_state()

  def load_state(self, state):
    if self.rescaler is None:
      self.predictor.load_state(state)
    else:
      self.predictor.load_state(state[0])
      self.rescaler.load_state_dict(state[1])


def compute_class_prior(labels, n_classes):
  zeros = np.zeros(n_classes)
  np.add.at(zeros, labels, 1)
  return torch.as_tensor(np.log(zeros / len(labels)), dtype=torch.float32)


class ClfArgminEnsemble(Predictor):
  """An ensemble of models, some of which my use a `ArgminTransformFunction` to
  fine tune their output.

  MCE is an ensemble of two such models, where one is the lower capacity model and is the
  higher capacity model, and both of which have an `ArgminTransformFunction` that
  applies an optimized classifier to the output of each model.
  """

  @classmethod
  def build(cls, version, params):
    # Backward compatibility
    if "post_process_logits" in params:
      del params["post_process_logits"]
    if "penalty" in params:
      if params["penalty"] is not None:
        raise NotImplementedError()
      del params["penalty"]
    return super().build(version, params)

  def __init__(self, heads: List[ClfHead], n_classes, add_prior=False,
               no_rescale_on_first_step=False):
    super().__init__()
    self.add_prior = add_prior
    self.n_classes = n_classes
    self.heads = nn.ModuleList(heads)

    self._head_names = [x.head_name for x in heads]
    self.register_buffer("class_prior", torch.zeros(1, n_classes))

    # Sometimes the first training step causes errors in the argmin operators due to
    # non-invertible hessians, presumably because the input features are random
    # This flag disables the rescaling for the first step
    self.no_rescale_on_first_step = no_rescale_on_first_step

  def reset_parameters(self):
    pass

  def preprocess_dataset(self, is_train, examples, dataset):
    if is_train:
      self.register_buffer(
        "class_prior", compute_class_prior([x.label for x in examples], self.n_classes).unsqueeze(0))
    for h in self.heads:
      h.predictor.preprocess_dataset(is_train, examples, dataset)

  def get_state(self):
    return [h.get_state() for h in self.heads]

  def load_state(self, state):
    for i, h in enumerate(self.heads):
      h.load_state(state[i])

  def get_features(self, batch, labels=None):
    out = []
    for h in self.heads:
      logits = h.predictor(batch, labels)
      out.append(logits.float())  # float in case we are in fp16 mode
    return out

  def compute_loss(self, features, labels):
    prior = self.class_prior if self.add_prior else None
    rescaled = []
    losses = []
    monitor = {}

    for logit, h in zip(features, self.heads):
      if h.rescaler is not None:
        if self.no_rescale_on_first_step and self.training:
          self.no_rescale_on_first_step = False
        else:
          logit, v = h.rescaler(logit, labels, base_x=prior)
        logit = F.log_softmax(logit, -1)

      if prior is not None:
        head_loss = ops.cross_entropy(logit+prior, labels)
      else:
        head_loss = ops.cross_entropy(logit, labels)
      monitor[h.head_name + "-loss"] = head_loss

      if h.nll_penalty is not None:
        losses.append(head_loss * h.nll_penalty)
      rescaled.append(logit)
    ensemble_logits = torch.stack(rescaled, 2).sum(2)
    if prior is not None:
      ensemble_logits += prior

    losses.append(ops.cross_entropy(ensemble_logits, labels))
    return TrainOutput(torch.stack(losses, 0).sum(), monitor)

  def compute_eval_output(self, batch):
    features = self.get_features(batch, None)
    prior = self.class_prior if self.add_prior else None
    rescaled = []
    for logit, h in zip(features, self.heads):
      if h.rescaler is not None:
        logit, v = h.rescaler(logit, None, base_x=prior)
      logit = F.log_softmax(logit, -1)
      rescaled.append(logit)
    return ClassifierEnsembleOutput(torch.stack(rescaled, -1), self._head_names, prior)

  def set_eval_features(self, features, labels):
    prior = self.class_prior if self.add_prior else None
    for logit, h in zip(features, self.heads):
      if h.rescaler is not None:
        logit = torch.cat(logit, 0)
        h.rescaler.set_eval_argmin_f(logit, labels, base_x=prior)

  def needs_eval_features(self):
    return any(x.rescaler is not None for x in self.heads)

  def has_batch_loss(self):
    return True

  def forward(self, input_features, label=None, mode="batch"):
    """
    This predictor has a "batch_loss", meaning its loss has to be computed using the entire batch.
    This means we can't split the batch onto different GPUs and compute the loss on each GPU
    independently.

    Instead, we have the predictor compute the loss in two steps, step one builds features that are
    computed for each example independently (and so can be done on different GPUs) and step 2
    computes the batch loss using the features, which must be done with the entire batch on
    one GPU. The `mode` argument switches out this method.

    This allows this class to be used with our `model.data_parallel.DataParallel` wrapper
    """
    if not self.training:
      return self.compute_eval_output(input_features)
    if mode == "batch":
      # Should be used if training on one GPU
      features = self.get_features(input_features, label)
      out = self.compute_loss(features, label)
      return out
    elif mode == "features":
      return self.get_features(input_features, label)
    elif mode == "loss":
      # input_feature should be from self(..., mode=features)
      return self.compute_loss(input_features, label)
    else:
      raise NotImplementedError(mode)


class ClfBiasMixinEnsemble(Predictor):
  """Implements the oracle baseline"""

  def __init__(self, embed_model: Layer, n_classes: int,
               bias: Predictor, embed_dim: int, entropy_w=None, do_mixin=True):
    super().__init__()
    self.bias = bias
    self.embed_model = embed_model
    self.embed_dim = embed_dim
    self.entropy_w = entropy_w
    self.do_mixin = do_mixin
    self.n_classes = n_classes
    self.clf = nn.Linear(embed_dim, n_classes)
    if do_mixin:
      self.mixin = nn.Linear(embed_dim, 1)
    else:
      self.mixin = None

  def reset_parameters(self):
    if self.mixin is not None:
      self.mixin.reset_parameters()
    self.clf.reset_parameters()

  def preprocess_dataset(self, is_train, examples, dataset):
    self.bias.preprocess_dataset(is_train, examples, dataset)

  def get_state(self):
    return [self.bias.get_state(),
            self.embed_model.state_dict(),
            self.clf.state_dict(),
            None if self.mixin is None else self.mixin.state_dict()]

  def load_state(self, state):
    self.bias.load_state(state[0])
    self.embed_model.load_state_dict(state[1])
    self.clf.load_state_dict(state[2])
    if self.mixin is not None:
      self.mixin.load_state_dict(state[3])

  def forward(self, features, labels=None, **kwargs):
    bias = self.bias(features, labels, **kwargs).float()
    bias = F.log_softmax(bias, -1)
    embed = self.embed_model(features, labels, **kwargs)
    main_pred = self.clf(embed).float()
    main_pred = F.log_softmax(main_pred, -1)

    if self.mixin is not None:
      scale = F.softplus(self.mixin(embed).float())
      scaled_bias = bias*scale
      joint = scaled_bias + main_pred
    else:
      joint = bias + main_pred
      scaled_bias = bias

    if isinstance(labels, (list, tuple)):
      labels, weights = labels
    else:
      weights = None

    if self.training:
      loss = ops.cross_entropy(joint, labels, weights)
      if self.entropy_w:
        logprob = F.log_softmax(scaled_bias, -1)
        prob = torch.exp(logprob)
        loss -= (prob*logprob).sum(1).mean() * self.entropy_w
      return TrainOutput(loss)
    else:
      return ClassifierEnsembleOutput(
        torch.stack([scaled_bias, main_pred, joint], -1),
        ["bias", "debiased", "joint"], None
      )


class ClfBiAdversary(Predictor):
  """Implements the adversary baseline"""

  def __init__(self, model, bias, n_classes, adv_w, bias_loss,
               main_loss=0, joint_loss=1.0, use_y_values=True,
               joint_adv=True):
    super().__init__()
    self.joint_adv = joint_adv
    self.joint_loss = joint_loss
    self.use_y_values = use_y_values
    self.main_loss = main_loss
    self.bias = bias
    self.model = model
    self.n_classes = n_classes
    self.adv_w = adv_w
    self.bias_to_main = nn.Linear(n_classes * (2 if use_y_values else 1), n_classes)
    self.main_to_bias = nn.Linear(n_classes * (2 if use_y_values else 1), n_classes)
    self.bias_loss = bias_loss

  def reset_parameters(self):
    self.bias_to_main.reset_parameters()
    self.main_to_bias.reset_parameters()

  def preprocess_dataset(self, is_train, examples, dataset):
    self.bias.preprocess_dataset(is_train, examples, dataset)
    self.model.preprocess_dataset(is_train, examples, dataset)

  def get_state(self):
    return [self.bias.get_state(),
            self.model.get_state(),
            self.bias_to_main.state_dict(),
            self.main_to_bias.state_dict()
            ]

  def load_state(self, state):
    self.bias.load_state(state[0])
    self.model.load_state(state[1])
    self.bias_to_main.load_state_dict(state[2])
    self.main_to_bias.load_state_dict(state[3])

  def forward(self, features, labels=None, **kwargs):
    bias = self.bias(features, labels, **kwargs).float()
    bias = F.log_softmax(bias, -1)

    main = self.model(features, labels, **kwargs).float()
    main = F.log_softmax(main, -1)

    joint = main + bias

    if isinstance(labels, (list, tuple)):
      labels, weights = labels
    else:
      weights = None

    if self.use_y_values:
      y_one_hot = ops.one_hot(labels, self.n_classes)
      bias_in = torch.cat([bias, y_one_hot], 1)
      main_in = torch.cat([main, y_one_hot], 1)
    else:
      bias_in = bias
      main_in = main

    b2m_w = self.bias_to_main.weight.float()
    b2m_b = self.bias_to_main.bias.float()
    m2b_w = self.main_to_bias.weight.float()
    m2b_b = self.main_to_bias.bias.float()

    main_pred = F.log_softmax(F.linear(bias_in.detach(), b2m_w, b2m_b), -1)
    bias_pred = F.log_softmax(F.linear(main_in.detach(), m2b_w, m2b_b), -1)

    if self.training:
      loss = ops.soft_cross_entropy(main_pred, main.detach())
      loss += ops.soft_cross_entropy(bias_pred, bias.detach())
      if self.adv_w:
        main_pred = F.log_softmax(F.linear(bias_in, b2m_w.detach(), b2m_b.detach()), -1)
        bias_pred = F.log_softmax(F.linear(main_in, m2b_w.detach(), m2b_b.detach()), -1)

        if self.joint_adv:
          loss -= ops.soft_cross_entropy(main_pred, main) * self.adv_w
          loss -= ops.soft_cross_entropy(bias_pred, bias) * self.adv_w
        else:
          # Don't backprop to the model that produced the targets
          loss -= ops.soft_cross_entropy(main_pred, main.detach()) * self.adv_w
          loss -= ops.soft_cross_entropy(bias_pred, bias.detach()) * self.adv_w

      bias_loss = ops.cross_entropy(bias, labels, weights)
      main_loss = ops.cross_entropy(main, labels, weights)
      joint_loss = ops.cross_entropy(joint, labels, weights)
      if self.main_loss:
        loss += self.main_loss * main_loss
      if self.bias_loss:
        loss += self.bias_loss * bias_loss
      if self.joint_loss:
        loss += self.joint_loss * joint_loss
      return TrainOutput(loss, {
        "bias-loss": bias_loss, "main-loss": main_loss, "joint-loss": joint_loss})

    return ClassifierEnsembleOutput(
      torch.stack([bias, main], -1),
      ["bias", "debiased"], None
    )

