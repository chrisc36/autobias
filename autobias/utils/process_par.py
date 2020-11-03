from multiprocessing import Lock
from multiprocessing import Pool
from typing import Iterable, List, Set

from tqdm import tqdm

from autobias.utils.py_utils import flatten_list, split, group


class Processor:

  def process(self, data: Iterable):
    """ Map elements to an unspecified intermediate format """
    raise NotImplementedError()


class AddableTuple:
  def __init__(self, *items):
    self.items = items

  def __add__(self, other):
    return AddableTuple(*[a + b for a, b in zip(self.items, other.items)])

  def __iter__(self):
    yield from self.items

  def __repr__(self):
    return tuple(self.items).__repr__()

  def __str__(self):
    return tuple(self.items).__str__()


def _process_and_count(data: List, preprocessor: Processor):
  count = len(data)
  output = preprocessor.process(data)
  return output, count


def process_par(data: List, preprocessor, n_processes=2,
                chunk_size=1000, name=None, already_grouped=False,
                initializer=None):
  if chunk_size <= 0:
    raise ValueError("Chunk size must be >= 0, but got %s" % chunk_size)
  if n_processes is not None and n_processes <= 0:
    raise ValueError("n_processes must be >= 1 or None, but got %s" % n_processes)
  n_processes = min(len(data), 1 if n_processes is None else n_processes)

  if n_processes == 1 and initializer is None:
    if already_grouped:
      data = data[0]
    return preprocessor.process(tqdm(data, desc=name, ncols=80))
  else:
    if not already_grouped:
      chunks = split(data, n_processes)
      chunks = flatten_list([group(c, chunk_size) for c in chunks])
      total = len(data)
    else:
      chunks = data
      total = sum(len(x) for x in data)
    pbar = tqdm(total=total, desc=name, ncols=80)
    lock = Lock()

    def call_back(results):
      with lock:
        pbar.update(results[1])

    with Pool(n_processes, initializer=initializer) as pool:
      results = [
        pool.apply_async(_process_and_count, [c, preprocessor], callback=call_back)
        for c in chunks
      ]
      results = [r.get()[0] for r in results]

    pbar.close()
    output = results[0]
    if output is not None:
      if isinstance(output, Set):
        for r in results[1:]:
          output |= r
      else:
        for r in results[1:]:
          output += r
    return output
