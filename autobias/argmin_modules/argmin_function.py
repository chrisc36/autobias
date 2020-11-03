import numpy as np
import torch
from scipy import optimize
from torch.autograd import Function

from autobias.utils import ops
from autobias.utils.configured import Configured


class ArgminFunction(Configured):
  """Function that computes v* = argmin_v f(v, x, w, c) and various associated methods"""

  def argmin_f(self, x, y, w=None, c=None) -> torch.Tensor:
    """Return the `v` that minimize `f` on the given inputs"""
    raise NotImplementedError()

  def apply_transform(self, x, v) -> torch.Tensor:
    """Apply a transformation that is associated with `argmin_f`

    Typically this computes predictions using input `x` and parameters `v`
    """
    raise NotImplementedError()

  def np_compute_f(self, x, y, v, w=None, c=None):
    """Compute f using numpy arrays

    Only used to do finite difference checks
    """
    return self.np_build_f_dv(x, y, w, c)(v)[0]

  def np_compute_f_dv(self, x, y, v, w=None, c=None):
    """Compute the gradients and output of f w.r.t. v using numpy arrays

    Only used to do finite difference checks
    """
    return self.np_build_f_dv(x, y, w, c)(v)[1]

  def np_build_f_dv(self, x, y, w=None, c=None):
    """Build a numpy function to compute the loss and gradient of f w.r.t. v

    Used as input for black-box optimization
    """
    raise NotImplementedError()

  def compute_f_dvdv_and_f_dvdx(self, x, y, v, w=None, c=None):
    """Compute the second-order gradients of f

    :return:
      f_dvdv a [n_parameter, n_parameter] tensor of the hessian of f w.r.t. v
      f_dvdx a ([n_parameter] + x.size()) tensor of the partial gradients of f w.r.t. x and v
    """
    raise NotImplementedError()

  def compute_argmin_f_dx(self, x, y, argmin_v, w=None, c=None):
    """Compute the gradient of `self.argmin_f` w.r.t. x

    :return:
      f_dx a [batch, n_features] tensor of the gradient of x w.r.t. argmin f(x, y, w)
    """
    raise NotImplementedError()

  @property
  def block_diagonal_hessian(self):
    """If true, f_dvdv would be of size [n_blocks, n, n] instead of [n_parameters, n_parameters]

    This indicates some of the parameters are computed independently of each other
    """
    raise NotImplementedError()

  def reset_parameters(self):
    """Resets any internal state (used to reset if warm-starting optimizer)"""
    pass


class JointArgminFunction(ArgminFunction):
  """Argmin function with a non-decomposable hessian"""

  def __init__(self, n_parameters, use_gesv=False):
    super().__init__()
    self.use_gesv = use_gesv
    self.n_parameters = n_parameters
    self.fn = self._build_fn()

  @property
  def block_diagonal_hessian(self):
    return False

  def _build_fn(self):
    # Function to compute the argmin output with backpropagation
    class Fn(Function):
      @staticmethod
      def forward(ctx, x, y, w=None, c=None):
        v, l = self._argmin_f(x, y, w, c)
        ctx.save_for_backward(x, y, v, w, c)
        if x.dtype == torch.float16:
          v = v.half()
        return v

      @staticmethod
      def backward(ctx, v_grad):
        if v_grad is None:
          return None, None, None, None

        if not torch.all(torch.isfinite(v_grad)):
          raise ValueError("v_grad NaN")

        x, y, opt_v, w, c = ctx.saved_tensors

        if ctx.needs_input_grad[-1]:
          raise NotImplementedError()

        dx = self.compute_argmin_f_dx(x, y, opt_v, w, c)

        if not torch.all(torch.isfinite(dx)):
          raise ValueError("dx NaN")

        dx = (v_grad.unsqueeze(1).unsqueeze(1) * dx).sum(0)

        return dx, None, None, None
    return Fn

  def _argmin_f(self, x, y, w=None, c=None):
    # Return the `v` that minimize `f` on the given inputs and the minimized loss
    # Doesn't need to support backpropagation
    raise NotImplementedError()

  def argmin_f(self, x, y, w=None, c=None) -> torch.Tensor:
    return self.fn.apply(x, y, w, c)

  def compute_argmin_f_dx(self, x, y, argmin_v, w=None, c=None):
    """
    :return: The ([n_parameter + x.size()) tensor for gradient of x w.r.t. to argmin_v(f(x, y, v, w)

    Follows: https://arxiv.org/abs/1607.05447
    """
    grads = self.compute_f_dvdv_and_f_dvdx(x, y, argmin_v, w, c)
    f_dvdv = grads[0]
    f_dvdx = torch.cat(grads[1:], 1)

    if not torch.all(torch.isfinite(f_dvdv)):
      self.compute_f_dvdv_and_f_dvdx(x, y, argmin_v, w, c)
      raise ValueError("f_dvdv NaN")

    if len(f_dvdv.size()) == 0 or f_dvdv.size(0) == 1:
      return -f_dvdx / f_dvdv
    else:
      n_classes, n_examples, n_features = f_dvdx.size()
      f_dvdx = f_dvdx.reshape(self.n_parameters, -1)

      if self.use_gesv:
        out = -torch.gesv(f_dvdx, f_dvdv)[0]
      else:
        inv = torch.inverse(f_dvdv)
        out = -torch.matmul(inv, f_dvdx)

      return out.view(self.n_parameters, n_examples, n_features)

  def __getstate__(self):
    state = dict(self.__dict__)
    del state["fn"]
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.fn = self._build_fn()


class NumpyOptimizer(Configured):
  def __init__(self, tol=None, warm_start=True, method=None):
    self.tol = tol
    self.warm_start = warm_start
    self._v0_warm = None
    self.method = method

  def copy(self):
    return NumpyOptimizer(self.tol, self.warm_start, self.method)

  def set_v0(self, v0):
    self._v0_warm = v0.cpu().numpy().astype(np.float64)

  def argmin_f(self, optimizable, *args, bounds=None):
    dtype = args[0].dtype
    device = args[0].device

    _args = []
    for x in args:
      if x is None:
        _args.append(x)
      else:
        x = ops.numpy(x)
        if x.dtype in (np.float32, np.float16):
          x = x.astype(np.float64)
        _args.append(x)
      args = _args

    fn = optimizable.np_build_f_dv(*args)
    result = optimize.minimize(
      fn, x0=self._v0_warm, bounds=bounds, jac=True,
      tol=self.tol, method=self.method)

    if self.warm_start:
      self._v0_warm = result.x
    return (torch.as_tensor(result.x, device=device, dtype=dtype),
            torch.as_tensor(result.fun, device=device, dtype=dtype))
