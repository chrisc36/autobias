import numpy as np
import torch
from scipy.special import logsumexp as np_logsumexp
from torch.nn import functional as F

from autobias.argmin_modules.argmin_function import JointArgminFunction, NumpyOptimizer
from autobias.utils import ops


class AffineNLL(JointArgminFunction):
  """
  Computes an affine transform of its inputs that optimizes cross-entropy loss on the
  given labels.
  """

  def __init__(self, n_features, n_classes, optimizer: NumpyOptimizer,
               bias=True, residual=False, penalty=None,
               fix_last_bias_to_zero=True):
    """
    :param n_features: Number of input features
    :param n_classes: Number of target classes
    :param optimizer: Optimizer to compute the argmin with
    :param bias: Include a bias
    :param residual: If n_feature == n_classes, apply the transform residually
    :param penalty: L2Norm penaly to include
    :param fix_last_bias_to_zero: Fix the bias of the last class to zero, this can help
                                  avoid non-invertible hessians
    """
    self.optimizer = optimizer
    self.n_features = n_features
    self.n_out = n_classes
    self.n_bias = 0 if not bias else n_classes - fix_last_bias_to_zero
    self.n_weights = self.n_out * self.n_features
    n_params = self.n_out * self.n_features + self.n_bias
    super().__init__(n_params)

    self.fix_last_bias_to_zero = fix_last_bias_to_zero

    self.residual = residual
    if residual and n_features != self.n_out:
      raise ValueError()

    self.fix_last_bias_to_zero = fix_last_bias_to_zero
    self.bias = bias
    self.n_classes = n_classes
    self.n_features = n_features
    self.penalty = penalty
    self.reset_parameters()

  def reset_parameters(self):
    self.optimizer.set_v0(torch.zeros(self.n_parameters))

  def _argmin_f(self, x, y, w=None, c=None):
    return self.optimizer.argmin_f(self, x, y, w, c)

  def _get_weights(self, v):
    if self.bias:
      return v[:self.n_weights].reshape(self.n_features, self.n_out)
    else:
      return v.reshape(self.n_features, self.n_out)

  def np_compute_f(self, x, y, v, w=None, c=None):
    scaled = self.np_get_logits(x, v, c)
    norms = np_logsumexp(scaled, 1)
    y_scores = scaled[np.arange(len(y)), y]
    nll = norms - y_scores

    if w is None:
      loss = nll.mean()
    else:
      loss = np.dot(nll, w) / w.sum()

    if self.penalty:
      loss += self.penalty.np_compute_loss(self._get_weights(v))

    return loss

  def np_compute_f_dv(self, x, y, v, w=None, c=None):
    if w is None:
      w_ex = 1
      w_sum = len(y)
    else:
      w_ex = np.expand_dims(w, 1)
      w_sum = w.sum()

    if self.fix_last_bias_to_zero:
      is_valid = slice(None)
      bias_is_valid = y < self.n_classes - 1
    else:
      is_valid = slice(None)
      bias_is_valid = slice(None)

    logits = self.np_get_logits(x, v, c)

    prob = ops.softmax_np(logits, 1)

    x_weighted = x*w_ex

    # [n_class, batch] * [batch, n_features] -> [n_class, n_features]
    grad = np.matmul(prob.T, x_weighted)
    np.add.at(grad, y[is_valid], -(x_weighted[is_valid]))

    grad = grad.T
    grad /= w_sum

    if self.penalty:
      grad += self.penalty.np_compute_loss_and_grad(self._get_weights(v))[1].reshape(grad.shape)

    grad = grad.ravel()

    if self.bias:
      if self.fix_last_bias_to_zero:
        bias_grad = (w_ex * prob[:, :-1]).sum(0)
      else:
        bias_grad = (w_ex * prob).sum(0)
      np.add.at(bias_grad, y[bias_is_valid], -1 if w is None else -w[bias_is_valid])
      bias_grad /= w_sum
      grad = np.concatenate([grad, bias_grad], 0)

    return grad

  def compute_f_dvdv_and_f_dvdx(self, x, y, v, w=None, c=None):
    # If I was doing this again, I would just use torch.gradient, but I ended up
    # writing this out by hand (accuracy confirmed with finite difference checks) and
    # this version is probably faster so I will leave it as is
    device = x.device
    n_out = self.n_out

    if w is not None:
      w_ex = w.unsqueeze(1)
      w_sum = w.sum()
    else:
      w_ex = torch.ones(1, dtype=x.dtype, device=device)
      w_sum = torch.as_tensor(y.size(0), dtype=x.dtype, device=device)

    n_features = self.n_features

    logits = self.apply_transform(x, v, c)
    probs = F.softmax(logits, -1)

    # ****************************************
    # *********** Computing f_dvdv ***********
    # ****************************************

    if self.bias:
      x_for_dvdv = F.pad(x, (0, 1, 0, 0), value=1)
      n_features += 1
    else:
      x_for_dvdv = x

    # einsum version:
    # f_dvdv = -torch.einsum("zk,zj,zn,zi->kjni", x_for_dvdv, probs, x_for_dvdv, probs)
    # other = torch.einsum("zj,zk,zi->jki", probs, x_for_dvdv, x_for_dvdv)
    # for j in range(self.n_out):
    #   f_dvdv[:, j, :, j] += other[j]

    prob_x = torch.unsqueeze(probs, 2) * torch.unsqueeze(x_for_dvdv, 1)
    prob_x = prob_x.view(len(y), -1)
    prob_x_t = (prob_x*w_ex).transpose(0, 1)
    f_dvdv = -torch.matmul(prob_x_t, prob_x).view(n_out, n_features, n_out, n_features)

    if not torch.all(torch.isfinite(f_dvdv)):
      raise ValueError("f_dvdv NaN")

    other = torch.matmul(prob_x_t, x_for_dvdv).reshape(n_out, n_features, n_features)

    for j in range(self.n_out):
      f_dvdv[j, :, j] += other[j]
    f_dvdv = f_dvdv.transpose(0, 1).transpose(2, 3)

    f_dvdv /= w_sum
    if self.penalty:
      if self.bias:
        weights = v[:self.n_weights].reshape(self.n_features, self.n_out)
        penalty = self.penalty.compute_hessian(weights)
        f_dvdv[:-1, :, :-1] += penalty.reshape(f_dvdv[:-1, :, :-1].size())
      else:
        f_dvdv += self.penalty.compute_hessian(v).reshape(f_dvdv.size())

    n = self.n_parameters + self.fix_last_bias_to_zero
    f_dvdv = f_dvdv.reshape(n, n)
    if self.fix_last_bias_to_zero:
      f_dvdv = f_dvdv[:-1, :-1]

    # ****************************************
    # *********** Computing f_dvdx ***********
    # ****************************************

    if self.bias:
      v_for_dvdx = v[:self.n_weights].clone()
    else:
      v_for_dvdx = v.clone()
    v_for_dvdx = v_for_dvdx.reshape(self.n_features, self.n_out)

    if self.residual:
      v_for_dvdx += torch.eye(self.n_out, dtype=v.dtype, device=device)

    # compute `grad_exp`
    v_t = v_for_dvdx.transpose(0, 1).unsqueeze(0).unsqueeze(2)
    x_t = x.transpose(0, 1).unsqueeze(1).unsqueeze(3)

    f_dvdx = v_t * x_t
    for k in range(self.n_features):
      f_dvdx[k, :, :, k] += 1

    # Add in `grad_z`
    grad_z = torch.matmul(probs, v_for_dvdx.transpose(0, 1))
    grad_z_x = x.transpose(0, 1).unsqueeze(2) * grad_z.unsqueeze(0)
    for k in range(self.n_features):
      f_dvdx[k, :] += -grad_z_x[k]

    if self.bias:
      # Add in the bias
      bias_grad = v_t.squeeze(0) - grad_z
      f_dvdx = torch.cat([f_dvdx, bias_grad.unsqueeze(0)], 0)

    f_dvdx *= probs.transpose(0, 1).unsqueeze(0).unsqueeze(3)

    # Backprop the y scores
    ix = torch.arange(0, len(y))
    for k in range(self.n_features):
      f_dvdx[k, y, ix, k] -= 1

    f_dvdx = f_dvdx.view(-1, x.size(0), self.n_features)
    if self.fix_last_bias_to_zero:
      f_dvdx = f_dvdx[:-1]

    # Apply per-example weighting
    if w is not None:
      f_dvdx *= w.unsqueeze(0).unsqueeze(2)
    f_dvdx /= w_sum

    return f_dvdv, f_dvdx

  def apply_transform(self, x, v, c=None):
    if self.bias:
      w = v[:self.n_weights]
    else:
      w = v

    out = torch.matmul(x, w.view(self.n_features, self.n_out))

    if self.bias:
      if self.fix_last_bias_to_zero:
        out[:, :-1] += v[-self.n_bias:]
      else:
          out += v[-self.n_bias:]

    if c is not None:
      out += c

    if self.residual:
      out += x

    return out

  def np_get_logits(self, x, v, c=None):
    if self.bias:
      w = v[:self.n_weights]
    else:
      w = v

    out = np.matmul(x, w.reshape(self.n_features, self.n_out))

    if self.bias:
      if self.fix_last_bias_to_zero:
        out[:, :-1] += np.expand_dims(v[-self.n_bias:], 0)
      else:
        out += np.expand_dims(v[-self.n_bias:], 0)

    if c is not None:
      out += c

    if self.residual:
      out += x
    return out

  def np_build_f_dv(self, x, y, w=None, c=None):
    if w is None:
      w_ex = 1
      w_sum = len(y)
    else:
      w_ex = np.expand_dims(w, 1)
      w_sum = w.sum()

    x_weighted = x * w_ex

    w_y_grad = np.zeros((self.n_out, self.n_features))
    np.subtract.at(w_y_grad, y, x_weighted)

    if self.bias:
      b_y_grad = np.zeros(self.n_bias)
      if self.fix_last_bias_to_zero:
        sl = y < self.n_classes - 1
      else:
        sl = None
      src = 1 if w is None else w[sl]
      target = y[sl]
      np.subtract.at(b_y_grad, target, src)
    ix = np.arange(len(y))

    def fn(v):
      logits = self.np_get_logits(x, v, c)
      y_scores = logits[ix, y]
      norms = np_logsumexp(logits, 1)

      if w is None:
        loss = (norms - y_scores).mean()
      else:
        loss = np.dot(norms - y_scores, w).sum() / w_sum

      prob = np.exp(logits - np.expand_dims(norms, 1))
      grad = w_y_grad + np.matmul(prob.T, x_weighted)

      grad = grad.T
      grad /= w_sum

      grad = grad.ravel()

      if self.penalty:
        p_loss, p_grad = self.penalty.np_compute_loss_and_grad(self._get_weights(v))
        grad += p_grad.reshape(grad.shape)
        loss += p_loss

      if self.bias:
        if self.fix_last_bias_to_zero:
          bias_grad = (w_ex * prob[:, :-1]).sum(0)
        else:
          bias_grad = (w_ex * prob).sum(0)
        bias_grad += b_y_grad
        bias_grad /= w_sum
        grad = np.concatenate([grad, bias_grad], 0)

      return loss, grad

    return fn
