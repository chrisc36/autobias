from collections import defaultdict

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
from torchvision import datasets

from autobias import config
from autobias.datasets.dataset import Dataset
from autobias.datasets.image_clf_example import ImageClfExample
from autobias.utils import py_utils


class AbstractMNISTWithBias(Dataset):
  def __init__(self, p, bias_name, is_train, per_class_slice, n_classes=None):
    self.p = p
    if n_classes is None:
      n_classes = 10
    self.n_classes = n_classes
    self.is_train = is_train
    self.per_class_slice = per_class_slice
    if self.per_class_slice:
      sample_str = "c" + str(per_class_slice[0]) + "-" + str(per_class_slice[1])
    else:
      sample_str = None
    super().__init__(
      "mnist-" + bias_name + "-" + str(p), "train" if is_train else "test", sample_str)

  def _load(self):
    mnist = datasets.MNIST(config.TORCHVISION_CACHE_DIR, self.is_train, download=True)
    prefix = "tr" if self.is_train else "te"

    per_class_ixs = [[] for _ in range(10)]  # class -> index in `mnist`
    for i, l in enumerate(mnist.targets):
      per_class_ixs[int(l)].append(i)

    # Consistent shuffle to pick the subsample
    state = np.random.RandomState(54689123)
    for l in per_class_ixs:
      state.shuffle(l)

    # Compile example in a list of `ImageClfExample`
    sl = slice(self.per_class_slice[0], self.per_class_slice[1])
    if self.per_class_slice[1] > max(len(x) for x in per_class_ixs):
      raise ValueError()
    all_examples = []
    for ix in py_utils.flatten_list(x[sl] for x in per_class_ixs):
      img, label = mnist[ix]
      all_examples.append(ImageClfExample(prefix + str(ix), img, label))

    # Filter for classes
    if self.n_classes:
      all_examples = [x for x in all_examples if x.label < self.n_classes]

    # Now add the bias, first we group by class
    class_to_img = defaultdict(list)
    for ex in all_examples:
      class_to_img[ex.label].append(ex)

    for _, class_examples in class_to_img.items():
      class_examples.sort(key=lambda x: x.example_id)
      # How many cases where the bias indicates the correct class
      n_correct = int(np.round(len(class_examples)*self.p))
      for i, ex in enumerate(class_examples):
        if i < n_correct:
          target_label = ex.label
        else:
          # Pick an incorrect class in a consistent manner
          state = np.random.RandomState(int(ex.example_id[2:])*1 + (1 if self.is_train else 2**20))
          target_label = (ex.label + state.randint(1, self.n_classes)) % self.n_classes
          if target_label == ex.label:
            raise RuntimeError()

        # Note the bias as an extra feature
        ex.other_features = dict(bias=target_label)

        # Let the sub-class modify the image in a way that reflects the bias
        self.modify_image(ex, target_label)
    return all_examples

  def modify_image(self, example, target_label):
    """Modify `example` by adding the bias indicating `target_label`"""
    raise NotImplementedError()


COLORS = [tuple(int(np.round(c * 255)) for c in color) for color in plt.cm.get_cmap('tab10').colors]


class MNISTBackgroundColor(AbstractMNISTWithBias):

  def __init__(self, p, is_train, per_class_slice, sz=None):
    self.sz = sz
    if sz is not None:
      name = "bg-color-sz" + str(sz)
    else:
      name = "bg-color"
    super().__init__(p, name, is_train, per_class_slice)

  def modify_image(self, ex, target_label):
    target_color = COLORS[target_label]
    gray_pixdata = ex.image.load()

    if self.sz is not None:
      sz = self.sz
      colored = Image.new("RGB", (28 + sz*2, 30 + sz*2))
      colored.paste(ex.image.convert("RGB"), (sz, sz))
      color_pixdata = colored.load()

      for y in range(0, 28 + sz*2):
        for x in range(0, 28 + sz*2):
          if x < sz or y < sz or x - sz >= 28 or x - sz >= 28 or gray_pixdata[x-sz, y-sz] < 100:
            color_pixdata[x, y] = target_color
    else:
      colored = ex.image.convert("RGB")
      color_pixdata = colored.load()

      for y in range(28):
        for x in range(28):
          if gray_pixdata[x, y] < 100:
            color_pixdata[x, y] = target_color
    ex.image = colored


class MNISTPatches(AbstractMNISTWithBias):
  def __init__(self, p, is_train, per_class_slice):
    super().__init__(p, "patches", is_train, per_class_slice)

  def modify_image(self, ex, target_label):
    target_color = COLORS[target_label]
    gray_pixdata = ex.image.load()
    color_img = ex.image.convert("RGB")
    resized = Image.new("RGB", (30, 30))
    resized.paste(color_img, (2, 2))
    color_pixdata = resized.load()
    patch_size = 6
    for patch_x in range(5):
      for patch_y in range(5):
        if patch_x == 0 and patch_y == 0:
          c = target_color
        else:
          c = COLORS[np.random.randint(0, 10)]
        for x in range(patch_x*patch_size, (patch_x+1)*patch_size):
          for y in range(patch_y*patch_size, (patch_y+1)*patch_size):
            if x < 2 or x >= 30 or gray_pixdata[x - 2, y - 2] == 0:
              color_pixdata[x, y] = c

    ex.image = resized


class MNISTDependent(AbstractMNISTWithBias):
  def __init__(self, p, is_train, per_class_slice):
    super().__init__(p, "corners", is_train, per_class_slice, 8)
    class_mod = 4
    self.class_mod = class_mod
    self.colors = COLORS[:self.class_mod]

  def modify_image(self, ex, target_label):
    gray_pixdata = ex.image.load()

    resized = Image.new("RGB", (30, 30))
    resized.paste(ex.image.convert("RGB"), (2, 2))
    color_pixdata = resized.load()

    ex.label = ex.label % self.class_mod
    target_label = target_label % self.class_mod
    ex.other_features["bias"] = target_label

    offset = target_label
    start = np.random.randint(0, self.class_mod)
    c1 = self.colors[start]
    c2 = self.colors[(start + offset) % self.class_mod]
    other = self.colors[np.random.randint(0, self.class_mod)]
    other = tuple(other)
    stripes = [
      (0, 10, c1),
      (10, 20, other),
      (20, 30, c2)
    ]
    for x0, x1, color in stripes:
      for x in range(x0, x1):
        for y in range(0, 30):
          if x < 2 or y < 2 or x >= 30 or y >= 30 or gray_pixdata[x - 2, y - 2] < 100:
            color_pixdata[x, y] = color

    ex.image = resized

