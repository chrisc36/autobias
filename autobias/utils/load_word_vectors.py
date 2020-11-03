"""Loading words vectors."""
import gzip
import logging
import pickle
from os.path import join, exists
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from autobias.config import WORD_VEC_DIR
from autobias.utils import downloader

# Can be set to keep loaded word vectors in memory, which is useful if we
# evaluating multiple models that use the same word vectors
GLOBAL_CACHE = None

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"


def download_word_vectors(vec_name):
  if vec_name == "crawl-300d-2M":
    if exists(join(WORD_VEC_DIR, "crawl-300d-2M.vec")):
      return
    downloader.download_zip("crawl-300d-2M.vec", FASTTEXT_URL, WORD_VEC_DIR)
  else:
    raise NotImplementedError()


def load_word_vectors(vec_name: str, vocab: Optional[Iterable[str]]=None,
                      n_words_to_scan=None, is_path=False):
  if vec_name == "random":
    if vocab is None:
      raise NotImplementedError()
    return list(vocab), [np.random.normal(size=(50,)) for _ in range(len(vocab))]

  download_word_vectors(vec_name)
  if not is_path:
    vec_path = join(WORD_VEC_DIR, vec_name)
  else:
    vec_path = vec_name
  if exists(vec_path + ".txt"):
    vec_path = vec_path + ".txt"
  elif exists(vec_path + ".txt.gz"):
    vec_path = vec_path + ".txt.gz"
  elif exists(vec_path + ".pkl"):
    vec_path = vec_path + ".pkl"
  elif exists(vec_path + ".vec"):
    vec_path = vec_path + ".vec"
  else:
    raise ValueError("No file found for vectors %s" % vec_name)
  return load_word_vector_file(vec_path, vocab, n_words_to_scan)


def load_word_vector_file(vec_path: str, vocab: Optional[Iterable[str]]=None,
                          n_words_to_scan=None):
  if vocab is not None:
    vocab = set(vocab)

  if GLOBAL_CACHE is not None and vec_path in GLOBAL_CACHE and vocab is not None:
    cur = GLOBAL_CACHE[vec_path]
    if all(x in cur for x in vocab):
      logging.info("Load word vectors from cache")
      tmp = {k: v for k, v in cur.items() if k in vocab and v is not None}
      return list(tmp.keys()), list(tmp.values())

  # notes some of the large vec files produce utf-8 errors for some words, just skip them
  if vec_path.endswith(".pkl"):
    with open(vec_path, "rb") as f:
      return pickle.load(f)
  elif vec_path.endswith(".txt.gz"):
    handle = lambda x: gzip.open(x, 'r', encoding='utf-8', errors='ignore')
  else:
    handle = lambda x: open(x, 'r', encoding='utf-8', errors='ignore')

  if n_words_to_scan is None:
    if vocab is None:
      logging.info("Loading word vectors from %s..." % vec_path)
    else:
      logging.info("Loading word vectors from %s for voc size %d..." % (vec_path, len(vocab)))
  else:
    if vocab is None:
      logging.info("Loading up to %d word vectors from %s..." % (n_words_to_scan, vec_path))
    else:
      logging.info("Loading up to %d word vectors from %s for voc size %d..." % (n_words_to_scan, vec_path, len(vocab)))

  words = []
  vecs = []
  pbar = tqdm(desc="word-vec", ncols=100, total=n_words_to_scan)
  with handle(vec_path) as fh:
    for i, line in enumerate(fh):
      if n_words_to_scan is not None and i >= n_words_to_scan:
        break
      word_ix = line.find(" ")
      if i == 0 and " " not in line[word_ix+1:]:
        # assume a header row, such as found in the fasttext word vectors
        continue
      pbar.update(1)
      word = line[:word_ix]
      if (vocab is None) or (word in vocab):
        words.append(word)
        vecs.append(np.fromstring(line[word_ix+1:], sep=" ", dtype=np.float32))

  pbar.close()

  if GLOBAL_CACHE is not None:
    if vec_path not in GLOBAL_CACHE:
      GLOBAL_CACHE[vec_path] = {}
    cur = GLOBAL_CACHE[vec_path]
    for k, v in zip(words, vecs):
      cur[k] = v
    for k in vocab:
      if k not in cur:
        cur[k] = None

  return words, vecs
