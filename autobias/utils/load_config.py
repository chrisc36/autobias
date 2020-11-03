import inspect
import re
from os.path import dirname


# We need to have everything in the namespace in order to get class objects
# from the config files, so we are stuck just importing everything at the top
from torch.nn import ModuleList

from autobias.datasets.entailment_datasets import *
from autobias.datasets.imagenet_animals import *
from autobias.datasets.mnist import *
from autobias.model.mnli_model import *
from autobias.model.mnist_model import *
from autobias.model.resent_model import *
from autobias.modules.attention_layers import *
from autobias.modules.layers import *
from autobias.training.data_batcher import *
from autobias.training.data_iterator import *
from autobias.modules.classifier_ensembles import *
from autobias.modules.image_transforms import *
from autobias.training.evaluator import *
from autobias.training.hard_easy_evaluator import *
from autobias.training.optimizer import *
from autobias.training.post_fit_rescaling_ensemble import FitRescaleParameters
from autobias.training.trainer import *
from autobias.argmin_modules.argmin_function import *
from autobias.argmin_modules.argmin_transform import *
from autobias.argmin_modules.l2_norm import *
from autobias.argmin_modules.affine_nll import *
from autobias.argmin_modules.multi_sigmoid_nlll import *


def _build_class_mapper(out):
  for name, obj in globals().items():
    if inspect.isclass(obj) and issubclass(obj, Configured):
      if obj.cls_name() in out:
        raise ValueError(obj.cls_name())
      out[obj.cls_name()] = obj


def _get_cls(class_name, class_mapping={}):
  if len(class_mapping) == 0:
    _build_class_mapper(class_mapping)
  if class_name in class_mapping:
    return class_mapping[class_name]
  else:
    raise NotImplementedError(class_name)


def load_config_json(js):
  if isinstance(js, dict):
    if "name" in js:
      params = {k: load_config_json(v) for k, v in js.items() if k not in ["name", "version"]}
      cls = _get_cls(js["name"])
      return cls.build(js.get("version", 0), params)
    else:
      return {k: load_config_json(v) for k, v in js.items()}
  elif isinstance(js, list):
    return [load_config_json(x) for x in js]
  else:
    return js


def load_config(filename: str):
  with open(filename, "r") as f:
    js = json.load(f)
  return load_config_json(js)


def load_model(source: str, best=False):
  logging.info("Loading model %s" % source)
  if source.endswith("/"):
    source = source[:-1]
  model: Model = load_config(join(dirname(source), "model.json"))

  if torch.cuda.is_available():
    map_location = None
  else:
    map_location = torch.device('cpu')

  if best:
    state = torch.load(join(source, BEST_WEIGHTS_NAME), map_location=map_location)
  elif exists(join(source, WEIGHTS_NAME)):
    state = torch.load(join(source, WEIGHTS_NAME), map_location=map_location)
  else:
    state = torch.load(join(source, "pytorch_model.bin"), map_location=map_location)

  if "state_dict" in state:
    pruned_state = {}
    for k, v in state["state_dict"].items():
      k = re.sub(r"\.module_list\.([0-9]+)\.", ".mappers.\\1.", k)
      pruned_state[k] = v
    state["state_dict"] = pruned_state
  model.load_state(state)
  return model
