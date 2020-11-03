from os import mkdir, listdir, makedirs
from os.path import join, exists
from shutil import rmtree

from autobias.utils.configured import config_to_json


def init_model_dir(output_dir, trainer, model):
  clear_if_nonempty(output_dir)
  makedirs(output_dir, exist_ok=True)
  with open(join(output_dir, "model.json"), "w") as f:
    f.write(config_to_json(model, 2))
  with open(join(output_dir, "trainer.json"), "w") as f:
    f.write(config_to_json(trainer, 2))


def select_subdir(output_dir):
  i = 0
  while True:
    candidate = join(output_dir, "r" + str(i))
    if not exists(candidate):
      try:
        mkdir(candidate)
        return candidate
      except FileExistsError:
        pass
    i += 1


def clear_if_nonempty(output_dir):
  if output_dir:
    if exists(output_dir) and listdir(output_dir):
      if input("%s is non-empty, override (y/n)?" % output_dir).strip() == "y":
        rmtree(output_dir)
      else:
        raise ValueError(
          "Output directory ({}) already exists and is not empty.".format(output_dir))
