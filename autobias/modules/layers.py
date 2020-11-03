import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm as add_weight_norm

from autobias.utils import ops
from autobias.utils.configured import Configured, get_configuration


class Layer(nn.Module, Configured):

  def reset_parameters(self):
    """Reset the parameter of this layer, this is NOT applied
    recursively so Layers should reset their children as well"""
    raise NotImplementedError(self.__class__.__name__)


class Mapper(Layer):
  def forward(self, x, mask=None):
    raise NotImplementedError()


class MapperSeq(Mapper):
  def __init__(self, mappers):
    super().__init__()
    self.mappers = nn.ModuleList(mappers)

  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    for i, m in enumerate(self.mappers):
      x = m(x, mask)
    return x


def seq(*layers):
  return MapperSeq(layers)


class NullMapper(Mapper):
  def forward(self, x, mask=None):
    return x

  def reset_parameters(self):
    pass


class Dropout(Mapper):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = dropout

  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    return F.dropout(x, self.dropout, self.training)


class VariationalDropout(Mapper):
  def __init__(self, dropout):
    super().__init__()
    self.dropout = dropout

  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    mask = F.dropout(torch.ones(x.size(0), x.size(2), dtype=x.dtype, device=x.device), self.dropout, self.training, True)
    return x * mask.unsqueeze(1)


class FullyConnected(Mapper):
  def __init__(self, in_dim, out_dim, fn="relu", use_bias=True,
               weight_norm=False):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.use_bias = use_bias
    self.weight_norm = weight_norm
    self.fn = fn
    self._fn = ops.get_fn(fn)
    self.layer = nn.Linear(in_dim, out_dim, use_bias)
    if self.weight_norm:
      self.layer = add_weight_norm(self.layer, dim=None)

  def reset_parameters(self):
    self.layer.reset_parameters()

  def forward(self, x, mask=None):
    return self._fn(self.layer(x))


class AveragePooling2d(Layer):
  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    return torch.mean(x, [2, 3])


class MaxPoolingNd(Layer):
  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    batch, c = x.size()[:2]
    return torch.max(x.view(batch, c, -1), 2)[0]


class Conv1D(Mapper):
  def __init__(self, input_dim, out_dim, width, fn="relu"):
    super().__init__()
    self.input_dim = input_dim
    self.width = width
    self.out_dim = out_dim
    if self.width % 2 != 1:
      raise ValueError()
    self.padding = self.width // 2
    self.weight = nn.Parameter(torch.Tensor(out_dim, input_dim, width))
    self.bias = nn.Parameter(torch.Tensor(out_dim))
    self.fn = fn
    self._fn = ops.get_fn(fn)

  def reset_parameters(self):
    self.bias.data.zero_()
    n = self.input_dim * self.width
    stdv = 1. / math.sqrt(n)
    self.weight.data.uniform_(-stdv, stdv)

  def forward(self, x, mask=None):
    x = ops.mask(x, mask).transpose(1, 2)
    out = F.conv1d(x, self.weight, self.bias, 1, self.padding, 1, 1)
    out = out.transpose(1, 2)
    return self._fn(out)


class Conv1x1(Layer):
  def __init__(self, fn, n_in, n_out):
    super().__init__()
    self.conv = nn.Conv2d(n_in, n_out, kernel_size=1, stride=1, bias=True)
    self.fn = fn
    self.n_in = n_in
    self.n_out = n_out
    self._fn = ops.get_fn(fn)

  def reset_parameters(self):
    self.conv.reset_parameters()

  def forward(self, x, mask=None):
    return self._fn(self.conv(x))


def _reset_lin(x):
  if isinstance(x, (nn.Linear, nn.Conv2d)):
    x.reset_parameters()


class Conv2d(Layer):
  def __init__(self, in_dim, out_dim, conv_kernel, fn="relu"):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    self.conv_kernel = conv_kernel
    self.fn = fn
    self.conv = nn.Conv2d(in_dim, out_dim, conv_kernel)
    self._fn = ops.get_fn(self.fn)

  def reset_parameters(self):
    self.conv.reset_parameters()

  def forward(self, x, mask=None):
    return self._fn(self.conv(x))


class MaxPooling(Layer):
  def forward(self, x, mask=None):
    return ops.max_pool(x, mask)

  def reset_parameters(self):
    pass


class AttentionPooling(Layer):
  def __init__(self, mapper, n_in, n_heads=1):
    super().__init__()
    self.n_in = n_in
    self.mapper = mapper
    self.n_heads = n_heads
    self.lin = nn.Linear(n_in, self.n_heads)

  def reset_parameters(self):
    self.lin.reset_parameters()

  def forward(self, x, mask=None):
    if self.mapper is not None:
      atten = self.lin(self.mapper(x, mask))
    else:
      atten = self.lin(x)  # [batch, seq_len, n_heads]

    atten = ops.mask_logits(atten, mask)

    if self.n_heads == 1:
      atten = F.softmax(atten.squeeze(2), 1)
      out = (x * atten.unsqueeze(2)).sum(1)
    else:
      atten = F.softmax(atten, 1)
      # [batch, dim, seq_len] * [batch, seq_len, n_heads] = [batch, dim, n_heads]
      out = torch.matmul(x.transpose(1, 2), atten)
      out = out.view(x.size(0), -1)

    return out


class SequenceToVector(Layer):
  def __init__(self, mapper, pooler, post_pooled):
    super().__init__()
    self.mapper = mapper
    self.pooler = pooler
    self.post_pooled = post_pooled

  def reset_parameters(self):
    pass

  def forward(self, x, mask=None):
    if self.mapper is not None:
      x = self.mapper(x, mask)
    x = self.pooler(x, mask)
    if self.post_pooled is not None:
      x = self.post_pooled(x, mask)
    return x


class ConcatAndMultFuse(Layer):
  def forward(self, vector1, vector2):
    return torch.cat([vector1 * vector2, vector1, vector2], 1)

  def reset_parameters(self):
    pass


