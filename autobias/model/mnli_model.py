from dataclasses import dataclass
from os import makedirs
from os.path import join, exists
from typing import List, Tuple, Any, Dict, Iterable, Union, Optional

import numpy as np
import torch
from transformers import BertTokenizer

from autobias import config
from autobias.datasets.dataset import Dataset
from autobias.datasets.entailment_datasets import load_hypothesis_bias, TextPairExample
from autobias.model.model import Model, Predictor
from autobias.modules.bert_layer import BertLayer, BertOutput
from autobias.modules.layers import Mapper, Layer
from autobias.modules.word_and_char_encoder import WordAndCharEncoder
from autobias.utils import ops, py_utils
from autobias.utils.ops import Mask, collate_flat_dict
from autobias.utils.process_par import Processor, process_par
from autobias.utils.tokenizer import NltkAndPunctTokenizer

"""
Modules and data pre-processing to train our BERT + Decomposable Attention ensemble, this ends 
up being quite a  pain since the models needs different tokenizations
"""


@dataclass
class TextPairTensors:
  a_embed: Union[List[torch.Tensor], torch.Tensor]
  b_embed: Union[List[torch.Tensor], torch.Tensor]
  a_mask: Mask
  b_mask: Mask
  other_features: Optional[Dict]

  def to(self, device):
    return TextPairTensors(
      [x.to(device) for x in self.a_embed],
      [x.to(device) for x in self.b_embed],
      self.a_mask.to(device),
      self.b_mask.to(device),
      None if self.other_features is None else {
        k: v.to(device) for k, v in self.other_features.items()}
    )

  def pin_memory(self):
    return TextPairTensors(
      [x.pin_memory() for x in self.a_embed],
      [x.pin_memory() for x in self.b_embed],
      self.a_mask.pin_memory(),
      self.b_mask.pin_memory(),
      None if self.other_features is None else {
        k: v.pin_memory() for k, v in self.other_features.items()}
    )


@dataclass
class DualTokenizedExample:
  """
  Text-pair example encoded both as BERT ids and as a pair of token sequences,
  """
  example_id: str

  # BERT data
  bert_input_ids: np.ndarray
  segment2_start: int

  # Token data
  a_tokens: Union[List[str], Tuple]
  b_tokens: Union[List[str], Tuple]

  label: Any
  other_features: Dict

  def get_len(self):
    return len(self.bert_input_ids)


@dataclass
class BertAndEmbedOutput:
  """
  `DualTokenizedExample` that has been embedded both by BERT and by word vectors
  """
  bert_embed: torch.Tensor
  bert_layers: List[torch.Tensor]
  bert_pooled: torch.Tensor
  bert_token_type_ids: torch.Tensor
  bert_attention_mask: torch.Tensor

  mask_a: Mask
  mask_b: Mask
  embed_a: torch.Tensor
  embed_b: torch.Tensor

  other_features: Dict

  def get_bert_output(self):
    return BertOutput(
      self.bert_embed, self.bert_layers, self.bert_pooled,
      self.bert_token_type_ids, self.bert_attention_mask, None)

  def get_text_pair_tensors(self):
    return TextPairTensors(
      self.embed_a, self.embed_b, self.mask_a, self.mask_b, None)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def convert_text_no_padding(tokenizer, max_seq_length, text_a, text_b=None):
  """Converts text to an input sequence suitable for BERT"""
  tokens_a = tokenizer.tokenize(text_a)

  tokens_b = None
  if text_b:
    tokens_b = tokenizer.tokenize(text_b)
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    len_a, len_b = len(tokens_a), len(tokens_b)
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    n_truncated = len_a - len(tokens_a), len_b - len(tokens_b)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > (max_seq_length - 2):
      n_truncated = (len(tokens_a) - (max_seq_length - 2), 0)
      tokens_a = tokens_a[:(max_seq_length - 2)]
    else:
      n_truncated = (0, 0)

  tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

  if tokens_b:
    text_b_start = len(tokens)
    tokens += tokens_b + ["[SEP]"]
    return tokenizer.convert_tokens_to_ids(tokens), text_b_start, n_truncated
  else:
    return tokenizer.convert_tokens_to_ids(tokens), None, n_truncated


class DualTokenizer(Processor):
  """Applies tokenizers to build `DualTokenizedExample` from `TextPairExample`"""

  def __init__(self, bert_tokenize, tokenizer, max_seq_len):
    self.bert_tokenize = bert_tokenize
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len

  def process(self, data: Iterable[TextPairExample]):
    out = []
    for x in data:
      if self.tokenizer is None:
        a, b = None, None
      else:
        a, b = self.tokenizer.tokenize(x.text_a), self.tokenizer.tokenize(x.text_b)
      ids, seg2, _ = convert_text_no_padding(self.bert_tokenize, self.max_seq_len, x.text_a, x.text_b)
      out.append(DualTokenizedExample(
        x.example_id, ids, seg2,
        a, b, x.label, x.other_features
      ))
    return out


def dual_tokenize_dataset(
    dataset, bert_tokenize: BertTokenizer,
    tokenizer: NltkAndPunctTokenizer, max_seq_len, n_processes):
  tok = DualTokenizer(bert_tokenize, tokenizer, max_seq_len)
  return process_par(dataset.load(), tok, n_processes, name="tokenizing")


class FromBertPredictor(Predictor):
  def __init__(self, predictor):
    super().__init__()
    self.predictor = predictor

  def reset_parameters(self):
    pass

  def has_batch_loss(self):
    return self.predictor.has_batch_loss()

  def forward(self, features, label=None, **kwargs):
    return self.predictor(features.get_bert_output(), label, **kwargs)


class ExtractPooled(Layer):

  def reset_parameters(self):
    pass

  def forward(self, features, label=None, **kwargs):
    return features.bert_pooled


class FromEmbeddingPredictor(Predictor):
  def __init__(self, predictor):
    super().__init__()
    self.predictor = predictor

  def reset_parameters(self):
    pass

  def has_batch_loss(self):
    return self.predictor.has_batch_loss()

  def forward(self, features, label=None, **kwargs):
    return self.predictor(features.get_text_pair_tensors(), label, **kwargs)


class FromBertPooled(Predictor):
  """Prediction that """
  def __init__(self, mapper):
    super().__init__()
    self.mapper = mapper

  def reset_parameters(self):
    pass

  def forward(self, bert_out: BertAndEmbedOutput, labels=None, **kwargs):
    return self.mapper(bert_out.bert_pooled)


class FromPooled(Predictor):
  def __init__(self, mapper):
    super().__init__()
    self.mapper = mapper

  def reset_parameters(self):
    pass

  def forward(self, bert_out: BertOutput, labels=None, **kwargs):
    return self.mapper(bert_out.pooled)


class NullClfPredictor(Predictor):
  """Always predicts a uniform distribution"""

  def __init__(self, n_out):
    self.n_out = n_out
    super().__init__()

  def reset_parameters(self):
      pass

  def forward(self, features: TextPairTensors, label=None, **kwargs):
    return torch.zeros(len(features.a_embed), self.n_out, device=features.a_embed.device)


class BifusePredictor(Predictor):
  def __init__(self, pre_mapper: Mapper, bifuse_layer,
               post_mapper: Mapper, pooler, pooled_mapper):
    super().__init__()
    self.pre_mapper = pre_mapper
    self.bifuse_layer = bifuse_layer
    self.post_mapper = post_mapper
    self.pooler = pooler
    self.pooled_mapper = pooled_mapper

  def reset_parameters(self):
    pass

  def forward(self, features: TextPairTensors, label=None, **kwargs):
    a_embed, a_mask = features.a_embed, features.a_mask
    b_embed, b_mask = features.b_embed, features.b_mask
    if self.pre_mapper:
      a_embed = self.pre_mapper(a_embed, a_mask)
      b_embed = self.pre_mapper(b_embed, b_mask)

    a_embed, b_embed = self.bifuse_layer(a_embed, b_embed, a_mask, b_mask)

    if self.post_mapper:
      a_embed = self.post_mapper(a_embed, a_mask)
      b_embed = self.post_mapper(b_embed, b_mask)

    pooled = torch.cat([self.pooler(a_embed, a_mask), self.pooler(b_embed, b_mask)], 1)
    pooled = self.pooled_mapper(pooled)
    return pooled


class MnliHypothesisOnlyBias(Predictor):
  def reset_parameters(self):
    pass

  def preprocess_dataset(self, is_train, examples, dataset):
    example_ids, model_out = load_hypothesis_bias(dataset.fullname)
    bias = {example_id: probs for example_id, probs in zip(example_ids, model_out.logprobs)}
    for ex in examples:
      if ex.other_features is None:
        ex.other_features = {}
      ex.other_features["bias"] = bias[ex.example_id]

  def forward(self, out, labels=None, **kwargs):
    return out.other_features["bias"]


class BertAndEmbedModel(Model):
  """Encodes text pairs as a `BertAndEmbedOutput`, which is passed to `self.predictor`"""

  def __init__(self, bert_model, max_seq_len, tokenizer, encoder: WordAndCharEncoder,
               predictor, needs_pooled=True):
    super().__init__()
    if predictor is None:
      raise ValueError()
    self.bert_model = bert_model
    self.max_seq_len = max_seq_len
    self._bert_tok = None
    self.tokenizer = tokenizer
    self.encoder = encoder
    self.bert = BertLayer(bert_model, pool=needs_pooled)
    self.predictor = predictor
    self._tok = None
    self.needs_pooled = needs_pooled

  def get_bert_tokenizer(self):
    if self._tok is None:
      self._tok = BertTokenizer.from_pretrained(self.bert_model)
    return self._tok

  def preprocess_datasets(self, datasets: List[Dataset], n_processes=None):
    tokenized_datasets: List[List[DualTokenizedExample]] = []
    for ds in datasets:
      do_cache = self.bert_model == "bert-base-uncased" and self.tokenizer is not None
      if do_cache:
        cache = join(config.DUAL_TOKENIZED_CACHE, ds.fullname + ".pkl")
        if exists(cache):
          tokenized_datasets.append(py_utils.load_pickle(cache))
          continue

      tokenized_datasets.append(dual_tokenize_dataset(
        ds, self.get_bert_tokenizer(), self.tokenizer, self.max_seq_len, n_processes))

      if do_cache:
        makedirs(config.DUAL_TOKENIZED_CACHE, exist_ok=True)
        py_utils.write_pickle(tokenized_datasets[-1], cache)

    if self.tokenizer is not None:
      voc = set()
      for ds in tokenized_datasets:
        for x in ds:
          voc.update(x.a_tokens)
          voc.update(x.b_tokens)

      self.encoder.set_vocab(voc)

      for ds in tokenized_datasets:
        for ex in ds:
          ex.a_tokens = self.encoder.tensorize(ex.a_tokens)
          ex.b_tokens = self.encoder.tensorize(ex.b_tokens)

    for tokenized, ds in zip(tokenized_datasets, datasets):
      self.predictor.preprocess_dataset(False, tokenized, ds)

    return tokenized_datasets

  def get_collate_fn(self):
    def collate(batch: List[DualTokenizedExample]):
      max_seq_len = max(len(x.bert_input_ids) for x in batch)
      sz = len(batch)
      input_ids = np.zeros((sz, max_seq_len), np.int64)
      segment_ids = torch.zeros(sz, max_seq_len, dtype=torch.int64)
      mask = torch.zeros(sz, max_seq_len, dtype=torch.int64)
      for i, ex in enumerate(batch):
        input_ids[i, :len(ex.bert_input_ids)] = ex.bert_input_ids
        segment_ids[i, :ex.segment2_start] = 1
        mask[i, :len(ex.bert_input_ids)] = 1

      input_ids = torch.as_tensor(input_ids)
      label_ids = torch.as_tensor(np.array([x.label for x in batch], np.int64))

      if self.encoder is not None:
        a_embed = ops.collate_list_of_tuples([x.a_tokens for x in batch])
        a_mask = ops.build_masks([len(x.a_tokens[0]) for x in batch])

        b_embed = ops.collate_list_of_tuples([x.b_tokens for x in batch])
        b_mask = ops.build_masks([len(x.b_tokens[0]) for x in batch])
      else:
        a_embed, a_mask, b_embed, b_mask = [None]*4

      if batch[0].other_features is not None and len(batch[0].other_features) > 0:
        other_features = collate_flat_dict([x.other_features for x in batch])
      else:
        other_features = {}

      return (input_ids, segment_ids, mask, a_embed, a_mask, b_embed, b_mask, other_features), label_ids

    return collate

  def forward(self, features, label=None, **kwargs):
    if self.predictor.has_batch_loss() and kwargs.get("mode") == "loss":
      return self.predictor(features, label, **kwargs)

    input_ids, segment_ids, mask, a_embed, a_mask, b_embed, b_mask, other_features = features
    bert_out: BertOutput = self.bert(input_ids, segment_ids, mask)

    if self.encoder is not None:
      a_embed = self.encoder(a_embed)
      b_embed = self.encoder(b_embed)
    else:
      a_embed, b_embed = None, None

    features = BertAndEmbedOutput(
      bert_out.embeddings, bert_out.layers, bert_out.pooled,
      bert_out.token_type_ids, bert_out.attention_mask,
      a_mask, b_mask, a_embed, b_embed, other_features
    )
    return self.predictor(features, label, **kwargs)

  def has_batch_loss(self):
    return self.predictor.has_batch_loss()

  def get_state(self):
    return (None if self.encoder is None else self.encoder.get_state()), \
           self.bert.state_dict(), \
           self.predictor.get_state()

  def load_state(self, state):
    if self.encoder is not None:
      self.encoder.load_state(state[0])
    self.bert.load_state_dict(state[1])
    self.predictor.load_state(state[2])
