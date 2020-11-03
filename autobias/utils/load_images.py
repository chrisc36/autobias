import os
import tempfile
from os import makedirs
from os.path import exists, join

from PIL import Image
from torchvision import transforms

from autobias.config import IMAGE_LOCAL_CACHE_RESIZED


def load_resized_img_with_cache(url, save_to_cache=True):
  if IMAGE_LOCAL_CACHE_RESIZED is None:
    return transforms.Resize(256)(Image.open(url))

  key = "-".join(url.split("/")[-2:])
  cache = join(IMAGE_LOCAL_CACHE_RESIZED, key)
  if not exists(cache):
    img = Image.open(url)
    img.load()
    img = transforms.Resize(256)(img)

    if save_to_cache:
      if not exists(IMAGE_LOCAL_CACHE_RESIZED):
        # Use exist_ok to avoid race conditions
        makedirs(IMAGE_LOCAL_CACHE_RESIZED, exist_ok=True)
      fd, tmp = tempfile.mkstemp(prefix="tmp-", dir=IMAGE_LOCAL_CACHE_RESIZED)
      os.close(fd)
      img.save(tmp, "JPEG")
      os.rename(tmp, cache)  # Use tmp a file -> rename to avoid race conditions
  else:
    img = Image.open(cache)
    img.load()

  img.load()

  return img