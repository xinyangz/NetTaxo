"""A utility class which wrapping C version of word2vec."""

import logging
import time
from pathlib import Path
from subprocess import call

import numpy as np

logger = logging.getLogger(__name__)


def load_word2vec_format(fname):
  """Load emebdding from word2vec text format.

  Args:
    fname: The file to load.

  Returns:
    emb: A dictionary from word to np.array embedding.
  """
  emb = dict()
  with open(fname, "r") as f:
    line = next(f)
    tok = line.strip().split()
    n_word, n_dim = int(tok[0]), int(tok[1])
    for line in f:
      tok = line.strip().split()
      word = tok[0]
      vec = np.array(list([float(t) for t in tok[1:]]))
      emb[word] = vec
  return emb


def save_word2vec_format(fname, emb):
  """Save embedding to word2vec text format.

  Args:
      fname: The file to save.
      emb: A dictionary from word to vector.
  """
  n_words = len(emb)
  n_dim = next(iter(emb.values())).shape[0]
  with open(fname, "w") as f:
    f.write(f"{n_words} {n_dim}\n")
    for word, vec in emb.items():
      vec_str = " ".join([str(num) for num in vec])
      f.write(f"{word} {vec_str}\n")


class SGNS_C:

  def __init__(self,
               dim=300,
               n_thread=8,
               window=10,
               negative=25,
               n_iter=5,
               lam_base=0.001,
               lam_net=1.,
               use_net=True,):
    self.syn0 = None
    self.syn1 = None
    self.syn0_base = None
    self.syn1_base = None
    self.syn0_inc = None
    self.syn1_inc = None
    self.dim = dim
    self.n_thread = n_thread
    self.window = window
    self.negative = negative
    self.n_iter = n_iter
    self.lam_base = lam_base
    self.lam_net = lam_net
    self.use_net = use_net

  def _assemble_emb(self, base, incremental):
    wv = dict()
    for word, vec in base.items():
      wv[word] = vec + incremental[word]
    return wv

  def load(self, syn0_file):
    self.syn0 = load_word2vec_format(syn0_file)
    logger.warn(f"Only syn0 is loaded from {syn0_file}")

  def train(self,
            corpus_file,
            net_corpus_file=None,
            alpha=None,
            syn0_base=None,
            syn1_base=None,
            syn0=None,
            syn1=None,
            syn1_net=None,
            output=None,
            incremental=False,
            n_thread=None,):
    if n_thread is not None:
      self.n_thread = n_thread
    w2v_file = Path("word2vec")
    if not w2v_file.is_file():
      raise FileNotFoundError("word2vec binary not found")
    inc = 1 if incremental else 0
    cmd = (f"./word2vec -pp 0 -train {corpus_file} -size {self.dim} -window "
           f"{self.window} -negative {self.negative} -iter {self.n_iter} "
           f"-lambda-base {self.lam_base} -threads {self.n_thread} -binary 0 "
           f"-incremental {inc} ")
    if alpha:
      cmd += f"-alpha {alpha} "
    # figure out output files
    if output is None:
      # TODO(xinyangz): we're overwritting these files!!
      # TODO(xinyangz): update word2vec
      assert syn0_base is not None
      assert syn1_base is not None
      assert syn0 is not None
      assert syn1 is not None
      syn0_base_out = syn0_base
      syn1_base_out = syn1_base
      syn0_out = syn0
      syn1_out = syn1
    else:
      syn0_base_out = f"{output}.syn0.base.txt"
      syn1_base_out = f"{output}.syn1.base.txt"
      syn0_out = f"{output}.syn0.txt"
      syn1_out = f"{output}.syn1.txt"
      syn1_net_out = f"{output}.syn1_net.txt"

    net = 1 if self.use_net and self.lam_net != 0 else 0
    if net_corpus_file is not None:
      cmd += f"-train-net {net_corpus_file} -net {net} -lambda-net {self.lam_net} "

    if syn0_base is not None:
      cmd += f"-syn0-base {syn0_base} "
    if syn1_base is not None:
      cmd += f"-syn1-base {syn1_base} "
    if syn0 is not None:
      cmd += f"-syn0 {syn0} "
    if syn1 is not None:
      cmd += f"-syn1 {syn1} "
    if syn1_net is not None:
      cmd += f"-syn1-net {syn1_net}"
    cmd += (f"-syn0-base-out {syn0_base_out} "
            f"-syn1-base-out {syn1_base_out} "
            f"-syn0-out {syn0_out} "
            f"-syn1-out {syn1_out} "
            f"-syn1-net-out {syn1_net_out}")

    logger.info("Start word2vec C training.")
    logger.info(f"Command: {cmd}")
    start = time.time()
    call(cmd, shell=True)
    end = time.time()
    logger.info(f"End word2vec C training. Time elapsed: {end - start}")

    syn0_base_emb = load_word2vec_format(syn0_base_out)
    syn1_base_emb = load_word2vec_format(syn1_base_out)
    syn0_emb = load_word2vec_format(syn0_out)
    syn1_emb = load_word2vec_format(syn1_out)
    self.syn0_base = syn0_base_emb
    self.syn1_base = syn1_base_emb
    self.syn0_inc = syn0_emb
    self.syn1_inc = syn1_emb
    self.syn0 = self._assemble_emb(syn0_base_emb, syn0_emb)
    self.syn1 = self._assemble_emb(syn1_base_emb, syn1_emb)

