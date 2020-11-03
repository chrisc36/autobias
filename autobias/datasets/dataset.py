"""General Dataset API"""
from typing import List

from autobias.utils import configured


class Dataset(configured.Configured):

  def __init__(self, domain, split, sample_name=None, cache=False):
    self.domain = domain
    self.split = split
    self.sample_name = sample_name
    self.cache = cache
    self._cache_data = None

  @property
  def fullname(self):
    """Return the dataset name.

    We use the `self.fullname` for caching in a few places, so it should uniquely identify
    the a set of examples the dataset loads"""
    name = self.domain + "-" + self.split
    if self.sample_name:
      return name + "-" + self.sample_name
    else:
      return name

  def _load(self) -> List:
    raise NotImplementedError()

  def load(self):
    if self.cache:
      if self._cache_data is None:
        self._cache_data = self._load()
      return self._cache_data
    return self._load()

