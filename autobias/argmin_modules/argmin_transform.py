import torch

from autobias.argmin_modules.argmin_function import ArgminFunction
from autobias.modules.layers import Layer
from torch.nn import functional as F


class ArgminTransformFunction(Layer):
  """Wraps and an `ArgminFunction` and applys it is as a nn.Module

  At test time we need to use the output `v` of the ArgminFunction
  on a sample of training data. This value is set with `self.set_eval_argmin_f`
  """

  def __init__(self, fn: ArgminFunction, backprop_argmin=True):
    super().__init__()
    self.backprop_argmin = backprop_argmin
    self.fn = fn
    self.register_buffer("_v_for_eval", torch.zeros(self.fn.n_parameters))

  def reset_parameters(self):
    self.fn.reset_parameters()

  def set_eval_argmin_f(self, x, y, w=None, base_x=None):
    with torch.no_grad():
      x = F.log_softmax(x, -1)
      self.register_buffer("_v_for_eval", self.fn.argmin_f(x, y, w, base_x))

  def forward(self, x, y, w=None, base_x=None):
    x = F.log_softmax(x, -1)
    if self.training:
      if self.backprop_argmin:
        v = self.fn.argmin_f(x, y, w, base_x)
      else:
        with torch.no_grad():
          v = self.fn.argmin_f(x, y, w, base_x)
    else:
      if self._v_for_eval is None:
        raise ValueError("Eval v not set")
      v = self._v_for_eval
    return self.fn.apply_transform(x, v), v


