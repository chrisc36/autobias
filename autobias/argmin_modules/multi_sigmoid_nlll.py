import numpy as np
import torch
from torch import nn
from torch.autograd import Function

from autobias.argmin_modules.argmin_function import ArgminFunction
from autobias.utils import ops


def softplus_np(x):
  return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def np_log_sigmoid(x):
  norm = softplus_np(-x)
  return -norm, -x - norm


# Its a bit of a hack to make this a `nn.Module` since it does not have a forward method, but it
# needs to be one since it needs a buffer (`_mapping`) that is moved to the correct device
class SharpenSigmoidNLLRemapped(ArgminFunction, nn.Module):
  """Computes the scale/shift parameters for series of clustered sigmoid predictions that
  minimizes loss of the given labels.  """

  def __init__(self, mapping, optimizer, residual=False, l1=None):
    """
    :param mapping: Array where mapping[i] is the cluster id of class i
    :param optimizer:
    :param residual:
    :param l1:
    """
    super().__init__()
    self.mapping = mapping
    n_classes = max(mapping) + 1
    self.n_parameters = n_classes*2
    self.n_classes = n_classes
    self.register_buffer("_mapping", torch.as_tensor(mapping, dtype=torch.long))
    self.mapping_np = np.array(mapping, dtype=np.int32)

    self.residual = residual
    self.optimizer = optimizer.copy()
    self.l1 = l1
    self.reset_parameters()
    self.fn = self._build_fn()

  @property
  def block_diagonal_hessian(self):
    return True  # Each cluster is optimized independently, and so has independent parameters

  @property
  def is_scalar(self):
    return False

  def _build_fn(self):
    # This module needs a custom gradient, which itself might depend the on the hyperparamters
    # set for this model, so we build a local autograd.Function
    class Fn(Function):
      @staticmethod
      def forward(ctx, x, y, w=None, c=None):
        v = self._argmin_f(x, y, w, c)[0]
        ctx.save_for_backward(x, y, v, w, c)
        return v

      @staticmethod
      def backward(ctx, v_grad):
        if v_grad is None:
          return None, None, None, None

        x, y, opt_v, w, c = ctx.saved_tensors
        if opt_v.dtype == torch.float64:
          x = x.double()
          w = None if w is None else w.double()
          c = None if c is None else c.double()
          v_grad = v_grad.double()

        if ctx.needs_input_grad[-1]:
          raise NotImplementedError()

        dx = self.compute_argmin_f_dx(x, y, opt_v, w, c)
        n_features, param_per_feature, batch = dx.size()
        v_grad = v_grad.reshape(self.n_classes, param_per_feature)[self._mapping]
        dx = (dx * v_grad.unsqueeze(2)).sum(1).transpose(0, 1)

        dx = dx.float()
        return dx, None, None, None
    return Fn

  def argmin_f(self, x, y, w=None, c=None):
    return self.fn.apply(x, y, w, c)

  def np_expand_dvdx(self, f_dvdx):
    block_size, batch, features = f_dvdx.shape

    expanded = np.zeros((block_size * features, batch, features))
    for i in range(features):
      start = block_size * i
      end = block_size * (i + 1)
      expanded[start:end, :, i] = f_dvdx[:, :, i]

    expanded = expanded.reshape((len(self.mapping_np), 2, batch, features))
    out = np.zeros((self.n_classes, 2, batch, features))
    np.add.at(out, self.mapping_np, expanded)
    return out.reshape(self.n_classes*2, batch, features)

  def compute_argmin_f_dx(self, x, y, argmin_v, w=None, c=None):
    grads = self.compute_f_dvdv_and_f_dvdx(x, y, argmin_v, w, c)
    f_dvdv = grads[0]
    f_dvdx = torch.cat(grads[1:], 1)

    # [n_features, 2, batch] for each feature, the deriviate w.r.t the w/b that was applied to it
    f_dvdx = f_dvdx.permute(2, 0, 1)

    # [n_classes, 2, 2] for each the inverse of hessian of w/b
    f_dvdv_inv = torch.inverse(f_dvdv)

    # [n_features, 2, 2] mapped to each feature
    f_dvdv_inv = f_dvdv_inv[self._mapping]

    # [n_features, 2, 2] * [n_features, 2, batch] = [n_features, 2, batch]
    return -torch.matmul(f_dvdv_inv, f_dvdx)

  def reset_parameters(self):
    self.optimizer.set_v0(torch.zeros(self.n_parameters))

  def _argmin_f(self, x, y, w=None, c=None):
    return self.optimizer.argmin_f(self, x, y, w, c)

  def np_compute_f(self, x, y, v, w=None, c=None):
    lp, one_minus_lp = np_log_sigmoid(self.np_get_logits(x, v, c))

    loss = one_minus_lp * (1 - y) + y * lp
    if w is None:
      loss = -loss.mean()
    else:
      loss = -np.dot(loss.mean(1), w) / w.sum()

    if self.l1:
      loss += np.abs(v[::2]).mean() * self.l1

    return loss

  def np_compute_f_dv(self, x, y, v, w=None, c=None):
    prob = ops.sigmoid_np(self.np_get_logits(x, v, c))
    diff = prob - y

    if w is not None:
      diff *= np.expand_dims(w, 1)
      grad = np.stack([diff * x, diff], -1).sum(0) / (w.sum() * len(self.mapping_np))
    else:
      grad = np.stack([diff * x, diff], -1).mean(0) / len(self.mapping_np)

    unmapped_grad = np.zeros((self.n_classes, 2), dtype=x.dtype)
    np.add.at(unmapped_grad, self.mapping_np, grad)
    grad = unmapped_grad

    if self.l1:
      grad[:, 0] += np.sign(v[::2]) * self.l1 / self.n_classes

    return grad.ravel()

  def np_build_f_dv(self, x, y, w=None, c=None):
    if w is not None:
      w_sum = w.sum()
      div = w_sum * len(self.mapping_np)
    else:
      div = len(self.mapping_np) * x.shape[0]

    def f(v):
      logits = self.np_get_logits(x, v, c)
      prob = ops.sigmoid_np(logits)
      diff = prob - y

      if w is not None:
        diff *= np.expand_dims(w, 1)

      grad = np.stack([diff * x, diff], -1).sum(0) / div

      lp, one_minus_lp = np_log_sigmoid(logits)
      loss = one_minus_lp * (1 - y) + y * lp
      if w is None:
        loss = -loss.mean()
      else:
        loss = -np.dot(loss.mean(1), w) / w.sum()

      unmapped_grad = np.zeros((self.n_classes, 2), dtype=x.dtype)
      np.add.at(unmapped_grad, self.mapping_np, grad)
      grad = unmapped_grad

      if self.l1:
        loss += np.abs(v[::2]).mean() * self.l1
        grad[:, 0] += np.sign(v[::2])*self.l1 / self.n_classes

      return loss, grad.ravel()
    return f

  def build_f_dv(self, x, y, w=None, c=None):
    if w is not None:
      w_sum = w.sum()
      div = w_sum * len(self.mapping_np)
    else:
      div = len(self.mapping_np) * x.shape[0]

    ixs = self._mapping.unsqueeze(1).repeat((1, 2))

    def f(v):
      logits = self.get_logits(x, v, c)
      prob = torch.sigmoid(logits)
      diff = prob - y

      if w is not None:
        diff *= w.unsqueeze(1)

      grad = torch.stack([diff * x, diff], -1).sum(0) / div

      loss = logits * (y - 1) + ops.log_sigmoid(logits)
      if w is None:
        loss = -loss.mean()
      else:
        loss = -(loss.mean(1)*w).sum() / w.sum()

      unmapped_grad = torch.zeros((self.n_classes, 2), dtype=x.dtype, device=x.device)
      unmapped_grad.scatter_add_(0, ixs, grad)
      grad = unmapped_grad

      if self.l1:
        loss += torch.abs(v[::2]).mean() * self.l1
        grad[:, 0] += torch.sign(v[::2])*self.l1 / self.n_classes

      return loss, grad.view(-1)
    return f

  def compute_f_dvdv_and_f_dvdx(self, x, y, v, w=None, c=None):
    device = x.device
    dtype = x.dtype

    logits = self.get_logits(x, v, c)
    probs = torch.sigmoid(logits)

    prob_mult = probs * (1 - probs)

    if w is not None:
      w_sum = w.sum()
      prob_mult_w = prob_mult * w.unsqueeze(1)
    else:
      prob_mult_w = prob_mult
      w_sum = torch.as_tensor(x.size(0), device=device, dtype=dtype)

    # print(self.n_classes)
    # raise ValueError()
    dvdv = torch.zeros(len(self.mapping_np), 2, 2, device=device, dtype=dtype)
    dvdv[:, 1, 1] = prob_mult_w.sum(0)

    dvdv[:, 1, 0] = (prob_mult_w*x).sum(0)
    dvdv[:, 0, 1] = dvdv[:, 1, 0]

    dvdv[:, 0, 0] = (prob_mult_w*x*x).sum(0)

    dvdv = ops.scatter_add_dim0(dvdv, self.n_classes, self._mapping)
    dvdv /= w_sum * x.size(1)

    tmp = prob_mult * (v.view(-1, 2)[:, 0][self._mapping] + self.residual)
    dvdx = torch.stack([x * tmp - y + probs, tmp], 0)
    if w is not None:
      dvdx *= w.unsqueeze(0).unsqueeze(2)

    dvdx /= w_sum * x.size(1)

    return dvdv, dvdx

  def np_get_logits(self, x, v, c):
    v = v.reshape(-1, 2)
    v = v[self.mapping_np]
    x_t = x * (v[:, 0] + self.residual) + v[:, 1]
    if c is not None:
      x_t += c
    return x_t

  def get_logits(self, x, v, c):
    v = v.view(-1, 2)
    v = v[self._mapping]
    out = x * (v[:, 0] + self.residual)
    out += v[:, 1]
    if c is not None:
      out += c
    return out

  def apply_transform(self, x, v):
    v = v.view(-1, 2)
    v = v[self._mapping]
    out = x * (v[:, 0] + self.residual)
    out += v[:, 1]
    return out
