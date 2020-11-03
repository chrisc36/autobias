"""Provides the `Configured` class.

The `Configured` class provides human readable serialization and version
numbering to subclasses.

We use this to save/load models and training specifications
"""

import json
from collections import OrderedDict
from inspect import signature, Parameter
from warnings import warn

import numpy as np
from torch import nn


class Configuration:

  def __init__(self, name, version, params):
    self.name = name
    self.version = version
    self.params = params

  def to_json(self, indent=None):
    return config_to_json(self, indent)


def _get_param_names(cls):
  """Returns all parameter names of the `__init__` method."""
  init = cls.__init__
  if init is object.__init__:
    return []  # No init args

  init_signature = signature(init)
  for param in init_signature.parameters.values():
    if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
      raise ValueError(cls.__name__ + " has kwargs or args in __init__")
  return [p for p in init_signature.parameters.keys() if p != "self"]


class Configured:
  """A class with a version number or with fields that are either `Configured`
  objects or 'simple' aka json-serializable types.

  Includes some methods for getting a json representation and for pickling.

  Subclasses should have constructor args with matching fields, it may
  have additional fields as long those fields only act as caches, subclasses
  are expected to be effectively stateless and treated as being immutable.

  These classes cam be used as a convenient way to store the specifications
  for models/datasets/metrics/optimizers in way that can be
  serialized to a human readable representation
  """

  version = 0

  @classmethod
  def build(cls, version, params):
    if cls.version != version:
      warn(("%s loaded with version %s, but class version is %s") %
           (cls.__name__, cls.version, version))
    try:
      return cls(**params)
    except TypeError as e:
      print(cls)
      print(params.keys())
      raise e

  @classmethod
  def _get_param_names(cls):
    return _get_param_names(cls)

  @property
  def name(self):
    return self.cls_name()

  @classmethod
  def cls_name(cls):
    return cls.__name__

  def get_config(self):
    return Configuration(self.name, self.version, self.get_params())

  def get_params(self):
    out = OrderedDict()
    for key in self._get_param_names():
      v = getattr(self, key)
      out[key] = get_configuration(v)
    return out

  def to_json(self, indent=None):
    return config_to_json(self, indent)


def get_configuration(obj):
  """Transform `obj` into a `Configuration` object or json-serialable type."""
  if isinstance(obj, Configured):
    return obj.get_config()

  obj_type = type(obj)

  if obj_type in (list, set, frozenset, tuple):
    return obj_type([get_configuration(e) for e in obj])
  elif obj_type == nn.ModuleList:
    # Hack so Configured object can take a list of modules as a input parameter,
    # and save them as a nn.ModuleList
    return [get_configuration(x) for x in obj]
  elif obj_type in (OrderedDict, dict):
    output = obj_type()
    for k, v in obj.items():
      if isinstance(k, Configured):
        raise ValueError()
      output[k] = get_configuration(v)
    return output
  elif obj_type in {str, int, float, bool, type(None), np.integer, np.floating, np.ndarray, np.bool}:
    return obj
  else:
    raise ValueError("Can't configure obj " + str(obj_type))


def _to_py(obj):
  if isinstance(obj, np.integer):
    return int(obj)
  elif isinstance(obj, np.floating):
    return float(obj)
  elif isinstance(obj, np.bool):
    return bool(obj)
  elif isinstance(obj, np.ndarray):
    return obj.tolist()
  else:
    return obj


class _ConfiguredJSONEncoder(json.JSONEncoder):
  """`JSONEncoder` that handles `configured` and `configuration` objects"""

  def default(self, obj):
    if isinstance(obj, Configuration):
      if "version" in obj.params or "name" in obj.params:
        raise ValueError()
      out = OrderedDict()
      out["name"] = obj.name
      if obj.version != 0:
        out["version"] = obj.version
      for k, v in obj.params.items():
        out[_to_py(k)] = _to_py(v)
      return out
    if isinstance(obj, np.integer):
      return int(obj)
    elif isinstance(obj, np.floating):
      return float(obj)
    elif isinstance(obj, np.bool_):
      return bool(obj)
    elif isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, Configured):
      return obj.get_config()
    else:
      return super(_ConfiguredJSONEncoder, self).default(obj)


def config_to_json(data, indent=None):
  # sort_keys=False since the configuration objects will dump their
  # parameters in an ordered manner
  return json.dumps(
      data, sort_keys=False, cls=_ConfiguredJSONEncoder, indent=indent)
