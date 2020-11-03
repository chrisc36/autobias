from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
from transformers import cached_path
from transformers.modeling_bert import BertEmbeddings, BertEncoder, \
  BertPooler, BertConfig, BERT_PRETRAINED_MODEL_ARCHIVE_MAP

from autobias import config
from autobias.config import TRANSFORMER_CACHE_DIR
from autobias.modules.layers import Layer
from autobias.utils.ops import load_and_log


@dataclass
class BertOutput:
  embeddings: torch.Tensor
  layers: List[torch.Tensor]
  pooled: torch.Tensor
  token_type_ids: torch.Tensor
  attention_mask: torch.Tensor
  other_features: Optional[Dict] = None


class BertLayer(Layer):
  """Wraps transformer's BERT modules as a `Layer`"""

  def __init__(self, bert_model: str, max_layer=None, pool=True, freeze_embeddings=False):
    super().__init__()
    self.freeze_embeddings = freeze_embeddings
    config = BertConfig.from_pretrained(bert_model, cache_dir=TRANSFORMER_CACHE_DIR)
    if max_layer is not None and not pool:
      config.num_hidden_layers = max_layer
    self.pool = pool
    self.max_layer = max_layer
    self.embeddings = BertEmbeddings(config)
    if config.num_hidden_layers > 0:
      self.encoder = BertEncoder(config)
      self.encoder.output_hidden_states = True
    else:
      self.encoder = None

    if pool:
      self.pooler = BertPooler(config)
    else:
      self.pooler = None
    self.config = config
    self.bert_model = bert_model

  def forward(self, input_ids, token_type_ids=None, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids)

    if attention_mask.dim() == 3:
      extended_attention_mask = attention_mask[:, None, :, :]

    elif attention_mask.dim() == 2:
      extended_attention_mask = attention_mask[:, None, None, :]

    else:
      raise ValueError()

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    dtype = self.embeddings.word_embeddings.weight.dtype
    extended_attention_mask = extended_attention_mask.to(
      dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if self.freeze_embeddings:
      with torch.no_grad():
        embedding_output = self.embeddings(input_ids, token_type_ids)
    else:
      embedding_output = self.embeddings(input_ids, token_type_ids)

    if self.encoder is not None:
      head_mask = [None] * self.config.num_hidden_layers
      encoded_layers = self.encoder(
        embedding_output, extended_attention_mask, head_mask=head_mask)[1]
    else:
      encoded_layers = []

    if not self.pool:
      return BertOutput(embedding_output, encoded_layers, None, token_type_ids, attention_mask)
    return BertOutput(embedding_output, encoded_layers, self.pooler(encoded_layers[-1]),
                      token_type_ids, attention_mask)

  def reset_parameters(self):
    archive_file = BERT_PRETRAINED_MODEL_ARCHIVE_MAP[self.bert_model]
    state_dict_file = cached_path(archive_file, cache_dir=config.TRANSFORMER_CACHE_DIR)
    state_dict = torch.load(state_dict_file,
                            map_location='cpu' if not torch.cuda.is_available() else None)

    pruned_state_dict = {}
    for key, value in state_dict.items():
      if key.startswith("cls"):
        continue

      if 'gamma' in key:
        key = key.replace('gamma', 'weight')
      if 'beta' in key:
        key = key.replace('beta', 'bias')
      if key.startswith("bert."):
        key = key[5:]

      pruned_state_dict[key] = value

    load_and_log(self, pruned_state_dict)
