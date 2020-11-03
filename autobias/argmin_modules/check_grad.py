import time

import numpy as np
import torch
from scipy import optimize
from scipy.linalg import block_diag
from torch import nn

from autobias.argmin_modules.affine_nll import AffineNLL
from autobias.argmin_modules.argmin_function import ArgminFunction, NumpyOptimizer
from autobias.argmin_modules.l2_norm import L2NormPenalty
from autobias.argmin_modules.multi_sigmoid_nlll import SharpenSigmoidNLLRemapped


def check_dv(input_fn, v_fn, fn):
  for i in range(10):
    inputs = input_fn()
    v = v_fn()

    def f_fn(v_in):
      return fn.np_compute_f(v=v_in, **inputs)

    f_ds_aprox = optimize.approx_fprime(v, f_fn, [1e-8])
    f_ds = fn.np_compute_f_dv(v=v, **inputs)
    if not np.allclose(f_ds_aprox, f_ds, atol=1e-6):
      raise ValueError()


def sanity_check_weighted(input_fn, v_fn, fn):
  for i in range(10):
    inputs = input_fn()
    inputs["v"] = v_fn()
    inputs["w"] = np.ones(len(inputs["x"]))
    unweighted_inputs = {k: v for k, v in inputs.items() if k != "w"}

    if not np.allclose(fn.np_compute_f(**inputs), fn.np_compute_f(**unweighted_inputs),
                       atol=1e-10):
      raise ValueError()

    if not np.allclose(fn.np_compute_f_dv(**inputs), fn.np_compute_f_dv(**unweighted_inputs),
                       atol=1e-10):
      raise ValueError()


def sanity_grad_weighted(input_fn, v_fn, fn: ArgminFunction):
  for i in range(10):
    inputs = input_fn()
    inputs["v"] = v_fn()
    inputs["w"][:] = 1
    inputs = to_tensor(inputs)
    input_with_w = inputs
    inputs_without_w = {k: v for k, v in inputs.items() if k != "w"}

    dvdv, dvdx = fn.compute_f_dvdv_and_f_dvdx(**input_with_w)
    dvdv_w, dvdx_w = fn.compute_f_dvdv_and_f_dvdx(**inputs_without_w)

    if not np.allclose(dvdv, dvdv_w, atol=1e-8):
      raise ValueError("Failed gradient weighted santiy check")

    input_with_w["w"][0] = 0
    inputs_without_w = {}
    for k, v in input_with_w.items():
      if k == "w" or v is None:
        continue
      elif k == "v":
        inputs_without_w[k] = v
      elif k == "c" and v.size(0) == 1:
        inputs_without_w[k] = v
      else:
        inputs_without_w[k] = v[1:]

    dvdv, dvdx = fn.compute_f_dvdv_and_f_dvdx(**inputs_without_w)
    dvdv_w, dvdx_w = fn.compute_f_dvdv_and_f_dvdx(**input_with_w)
    if not np.allclose(dvdv, dvdv_w, atol=1e-8):
      raise ValueError("Failed gradient weighted santiy check")


def to_tensor(dict_of_tensors, dtype=None):
  return {k: None if v is None else torch.as_tensor(v, dtype=dtype) for k, v in dict_of_tensors.items()}


def check_curried_grad(input_fn, v_fn, fn: ArgminFunction):
    # Sanity check the curried grad/loss functions, which might do some
    # additional caching, or other optimization
    for i in range(10):
      inputs = input_fn()
      v = v_fn()

      loss1 = fn.np_compute_f(v=v, **inputs)
      grad1 = fn.np_compute_f_dv(v=v, **inputs)

      try:
        loss2, grad2 = fn.np_build_f_dv(**inputs)(v)

        if not np.allclose(loss1, loss2, atol=1e-8):
          raise ValueError()

        if not np.allclose(grad1, grad2, atol=1e-8):
          raise ValueError()
      except NotImplementedError:
        pass


def check_dvdv(input_fn, v_fn, fn):
  for it in range(10):
    inputs = input_fn()
    v = v_fn()

    f_dvdv = fn.compute_f_dvdv_and_f_dvdx(v=torch.as_tensor(v), **to_tensor(inputs))[0].numpy()

    if fn.block_diagonal_hessian:
      f_dvdv = block_diag(*[x for x in f_dvdv])
    out = []
    for i in range(len(v)):
      def f_fn(v_in):
        return fn.np_compute_f_dv(v=v_in, **inputs)[i]
      out.append(optimize.approx_fprime(v, f_fn, 1e-8))
    f_dvdv_aprox = np.stack(out, 0)

    if not np.allclose(f_dvdv_aprox, f_dvdv, atol=1e-6):
      raise ValueError()


def check_dvdv_scalar(input_fn, v_fn, fn):
  for i in range(10):
    inputs = input_fn()
    v = v_fn()
    def f_fn(v_in):
      inputs["v"] = v_in
      return fn.np_compute_f_dv(**inputs)
    f_dvdv_aprox = optimize.approx_fprime(np.expand_dims(v, 0), f_fn, 1e-8)

    inputs["v"] = v
    f_dvdv = fn.compute_f_dvdv_and_f_dvdx(**to_tensor(inputs))[0].numpy()

    if not np.allclose(f_dvdv_aprox, f_dvdv, atol=1e-6):
      raise ValueError()


def check_dvdx(input_fn, v_fn, fn):
  for i in range(10):
    inputs = input_fn()
    x = inputs["x"]
    v = v_fn()
    out = []
    for j in range(len(v)):
      def f_fn(x_p):
        inputs["x"] = x_p.reshape(x.shape)
        return fn.np_compute_f_dv(v=v, **inputs)[j]

      out.append(optimize.approx_fprime(x.ravel(), f_fn, 1e-8).reshape(x.shape))
    f_dsdx_aprox = np.stack(out, 0)
    f_dsdx = fn.compute_f_dvdv_and_f_dvdx(v=torch.as_tensor(v), **to_tensor(inputs))[1].numpy()

    if fn.block_diagonal_hessian:
      f_dsdx = fn.np_expand_dvdx(f_dsdx)

    if f_dsdx_aprox.shape != f_dsdx.shape:
      raise ValueError(f"Aprox shape {f_dsdx_aprox.shape} != actual shape {f_dsdx.shape}")

    if not np.allclose(f_dsdx_aprox, f_dsdx, atol=1e-6):
      raise ValueError("dvdx Error %d" % i)


def sanity_check_gradients(input_fn, fn: ArgminFunction, step_size=1e-6):
  # Due to precision limitation, its hard to check the overal gradient with finite differences,
  # but we can at last santiy check it
  # Sometime `step_size` needs to be tuned
  n_greater = 0
  n_same = 0
  n_decreased = 0
  n_fail = 0
  iterations = 20
  for i in range(iterations):
    inputs = to_tensor(input_fn())
    inputs["x"] = nn.Parameter(inputs["x"])

    try:
      v = fn.argmin_f(**inputs)[0]

      loss = v.sum()
      loss.backward()
      x = inputs["x"]
      inputs["x"] = x + x.grad*step_size
      loss2 = fn.argmin_f(**inputs)[0].sum()
      if loss2 > loss:
        n_greater += 1
      elif loss2 == loss:
        n_same += 1
      else:
        n_decreased += 1
    except FloatingPointError as e:
      raise e

  print(f"Sanity check x, score increased {n_greater}, same {n_same}, decreased {n_decreased} fail {n_fail}")


def check_grad_argmin(n_examples, n_features, n_classes, weighted,
                      fn: ArgminFunction, labels="one-hot",
                      seed=None, add_c=False):
  np.seterr('raise')
  rng = np.random if seed is None else np.random.RandomState(seed)

  def get_inputs():
    if weighted:
      w = rng.uniform(0.3, 1, n_examples)
    else:
      w = None

    # Back sure there at least two of each class
    if labels == "soft":
      y = np.random.dirichlet([1]*n_classes, n_examples)
    elif labels == "one-hot":
      y = np.concatenate([
        np.arange(n_classes),
        np.arange(n_classes),
      ], 0)
      if n_examples > len(y):
        y = np.concatenate([
          y,
          rng.randint(0, n_classes, n_examples-len(y))
        ])
      elif n_examples < len(y):
        y = y[:n_examples]
    else:
      raise RuntimeError()

    x = rng.uniform(-1, 1, (n_examples, n_features))

    if add_c:
      c = rng.uniform(-0.3, 0.3, (n_examples, n_classes))
    else:
      c = None
    return dict(x=x, y=y, w=w, c=c)

  def get_v():
    return rng.uniform(-0.8, 0.8, fn.n_parameters)

  if weighted:
    sanity_check_weighted(get_inputs, get_v, fn)
  check_dv(get_inputs, get_v, fn)
  print("Passed gradient check")

  check_curried_grad(get_inputs, get_v, fn)
  print("Passed curried function check")

  if weighted:
    sanity_grad_weighted(get_inputs, get_v, fn)

  check_dvdv(get_inputs, get_v, fn)
  check_dvdx(get_inputs, get_v, fn)
  print("Passed second order gradient checks")

  sanity_check_gradients(get_inputs, fn)


def test_affine():
  n_classes = 3
  n_features = 3
  n_examples = 8

  opt = NumpyOptimizer(tol=1e-12)

  fn = AffineNLL(
    n_features, n_classes, opt,
    penalty=L2NormPenalty(0.01),
    residual=False, fix_last_bias_to_zero=True,
  )
  check_grad_argmin(
    n_examples, n_features, n_classes,
    False, fn, labels="one-hot", add_c=True)


def test_sigmoid():
  n_classes = 3
  n_features = 3
  n_examples = 8

  opt = NumpyOptimizer(tol=1e-12)

  fn = SharpenSigmoidNLLRemapped([0, 0, 1], opt, residual=True, l1=0.01)
  check_grad_argmin(
    n_examples, n_features, n_classes,
    False, fn, labels="soft", add_c=True)


if __name__ == "__main__":
  test_affine()