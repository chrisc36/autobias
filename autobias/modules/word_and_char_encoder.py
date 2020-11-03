from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F, init

from autobias.modules.layers import Layer
from autobias.utils import load_word_vectors

BOW_CHAR = 255
EOW_CHAR = 256
PAD_CHAR = 257
NUM_CHARS = PAD_CHAR+1


def words_to_char_ids_py(words, word_length):
  out = np.full((len(words), word_length), PAD_CHAR, np.int32)
  out[:, 0] = BOW_CHAR
  for i, word in enumerate(words):
    word = word.encode("utf-8")
    word = list(word)[:word_length-2]
    out[i, 1:len(word)+1] = word
    out[i, len(word)+1] = EOW_CHAR
  return out


class WordAndCharEncoder(Layer):
  """Layer that embeds text using word vectors and character embeddings.

  Client needs to class `set_vocab` on this class with the input vocab,
  should use the `tensorsize` method to convert tokens to tensors,
  and then pass those tensors into `forward` method to get the embeddings
  """
  def __init__(self, word_vectors, first_n=None, char_embed_dim=None, character_mapper=None,
               character_pooler=None, word_length=30):
    super().__init__()
    if character_mapper is not None and character_pooler is None:
      raise ValueError()
    if character_pooler is not None and character_mapper is None:
      raise ValueError()

    self.character_mapper = character_mapper
    self.char_embed_dim = char_embed_dim
    self.character_pooler = character_pooler
    self.word_vectors = word_vectors
    self.first_n = first_n
    self.word_length = word_length

    self._input_vocab = None

    self._word_to_ix = None

    if self.use_word_vecs:
      # Buffer since it is fixed
      self.register_buffer("_embeddings", None)

    if self.use_chars:
      self._char_embed = nn.Parameter(torch.Tensor(NUM_CHARS, char_embed_dim))
    else:
      self._char_embed = None

  def clear(self):
    if self.use_word_vecs:
      self._word_to_ix = None
      self.register_buffer("_embeddings", None)

  def reset_parameters(self):
    if self._char_embed is not None:
      init.normal_(self._char_embed, mean=0, std=0.2)

  def learn_vocab(self, train_word_counts, other_vocab):
    voc = set(train_word_counts).union(other_vocab)
    self.set_vocab(voc)

  @property
  def unk_ix(self):
    return 0

  @property
  def is_vocab_set(self):
    return True

  def get_state(self):
    return self.state_dict()

  def load_state(self, state):
    self.load_state_dict(state)

  def set_vocab(self, vocab):
    if self.use_word_vecs:
      if self._word_to_ix is None:
        # Load and cache the words vectors
        words, vecs = load_word_vectors.load_word_vectors(
          self.word_vectors, vocab, self.first_n)

        dim = len(vecs[0])
        self._word_to_ix = {w: i+1 for i, w in enumerate(words)}
        embed = torch.as_tensor(np.stack([np.zeros(dim, dtype=np.float32)] + vecs))
        self.register_buffer("_embeddings", embed)

      else:
        # Add the words to the existing examples # TODO do
        vocab = {w for w in vocab if w not in self._word_to_ix}
        if len(vocab) == 0:
          return
        words, vecs = load_word_vectors.load_word_vectors(
          self.word_vectors, vocab, self.first_n)
        if len(words) == 0:
          return
        for w in words:
          self._word_to_ix[w] = len(self._word_to_ix) + 1
        embed = self._embeddings
        new_embed = torch.cat([embed, torch.as_tensor(np.stack(vecs))], 0)
        self.register_buffer("_embeddings", new_embed)

  @property
  def use_word_vecs(self):
    return self.word_vectors is not None

  @property
  def use_chars(self):
    return self.char_embed_dim is not None

  def tensorize(self, tokens: List[str]):
    if self.use_word_vecs:
      w_ids = (np.array([self._word_to_ix.get(x, 0) for x in tokens], np.int32), )
    else:
      w_ids = ()

    if self.use_chars:
      c_ids = (words_to_char_ids_py(tokens, self.word_length), )
    else:
      c_ids = ()

    return w_ids + c_ids

  def forward(self, ids):
    embed = []

    if self.use_word_vecs:
      embed.append(F.embedding(ids[0], self._embeddings))

    if self.use_chars:
      c_ids = ids[self.use_word_vecs]
      c_embed = F.embedding(c_ids, self._char_embed)
      batch, seq_len, word_size, dim = c_embed.size()
      c_embed = c_embed.view(batch, seq_len*word_size, dim)
      c_embed = self.character_mapper(c_embed)
      c_embed = c_embed.view(batch, seq_len, word_size, -1)
      embed.append(self.character_pooler(c_embed))

    return torch.cat(embed, -1)

  def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs):
    # Don't load the embedding buffer if self._embeddings is not set,
    # it will be created by loading word vectors wgeb `set_vocab` is called
    if prefix + "_embeddings" in state_dict and self._embeddings is None:
      del state_dict[prefix + "_embeddings"]
    super()._load_from_state_dict(
      state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs)