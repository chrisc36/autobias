import logging
from typing import List

import nltk
import regex

from autobias.utils.configured import Configured
from autobias.utils.py_utils import flatten_list

_double_quote_re = regex.compile(u"\"|``|''|--")


class NltkAndPunctTokenizer(Configured):
  def __init__(self):
    self.split_dash = True
    self.split_single_quote = False
    self.split_period = False
    self.split_comma = False

    # Unix character classes to split on
    resplit = r"\p{Pd}\p{Po}\p{Pe}\p{S}\p{Pc}"

    # A list of optional exceptions, for this character will we trust nltk
    # to split correctly
    dont_split = ""
    if not self.split_dash:
      dont_split += "\-"
    if not self.split_single_quote:
      dont_split += "'"
    if not self.split_period:
      dont_split += "\."
    if not self.split_comma:
      dont_split += ","

    resplit = "([" + resplit + "]|'')"
    if len(dont_split) > 0:
      split_regex = r"(?![" + dont_split + "])" + resplit
    else:
      split_regex = resplit

    self.split_regex = regex.compile(split_regex)
    try:
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
    except LookupError:
      logging.info("Downloading NLTK punkt tokenizer")
      nltk.download('punkt')
      self.sent_tokenzier = nltk.load('tokenizers/punkt/english.pickle')
    self.word_tokenizer = nltk.TreebankWordTokenizer()

  """Tokenize ntlk, but additionally split on most punctuations symbols"""

  def retokenize(self, x):
    if _double_quote_re.match(x):
      # Never split isolated double quotes(TODO Just integrate this into the regex?)
      return (x, )
    return [x.strip() for x in self.split_regex.split(x) if len(x) > 0]

  def tokenize(self, text: str) -> List[str]:
    return flatten_list(flatten_list(self.retokenize(w)
                                     for w in self.word_tokenizer.tokenize(s))
                        for s in self.sent_tokenzier.tokenize(text))

