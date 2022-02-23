"""Data structures for taxonomy and nodes in a taxonomy."""

from __future__ import annotations

import collections
import gzip
import json
import logging
import pathlib as plib
from typing import Dict, List, Optional

import pydot

import utils
from hin import HIN

logger = logging.getLogger(__name__)

LevelEmbed = collections.namedtuple("LevelEmbed", "word net")


def parse_prefix(prefix):
  return list(map(int, prefix.split("/")))


class TaxoNode:
  """A node in a taxonomy."""

  def __init__(self,
               prefix: str,
               docs: Dict[str, List[str]],
               terms: List[str],
               doc_prior: Optional[Dict[str, float]] = None,
               term_prior: Optional[Dict[str, float]] = None,
               tf: Optional[Dict[str, float]] = None,
               idf: Optional[Dict[str, float]] = None,
               parent: Optional[TaxoNode] = None):
    # Each node is associate with a prefix string
    # indicating its position in the taxonomy
    # e.g. 0/1/0 is a 2nd level taxonomy node
    # root -> second child -> first child
    self.prefix = prefix

    # parent & child references
    self.parent = parent
    if parent is not None:
      parent.add_child(self)
    self.children: List[TaxoNode] = []

    # Each node is associated with a set of terms and a
    # set of documents.
    self.docs = docs
    self.terms = terms

    # Each node is also associated with weights on its documents
    # and terms.
    self.doc_prior = doc_prior
    self.term_prior = term_prior
    self.term_scores = dict()  # for final taxonomy

    self._num_child = None
    self._tf = tf
    self._idf = idf
    # self.clustering_probs = None
    # self.ranking = None

  def add_child(self, node):
    node.parent = self
    self.children.append(node)

  def set_parent(self, node):
    self.parent = node

  @property
  def num_child(self):
    return len(self.children)

  @property
  def dot_repr(self):
    terms_str = "\n".join(self.top_terms)
    ret = pydot.Node(self.prefix, shape="record", label="{%s|%s}" % (self.prefix, terms_str))
    return ret

  @dot_repr.setter
  def dot_repr(self, value):
    self._dot_repr = value

  @property
  def tf(self):
    if self._tf is not None:
      return self._tf
    elif (self.docs is not None) and (self.terms is not None):
      self._compute_tf_idf()
      return self._tf
    else:
      raise RuntimeError("Either documents or pre-computed tf-idf should be given")

  @tf.setter
  def tf(self, value):
    assert isinstance(value, dict)
    self._tf = value

  @property
  def idf(self):
    if self._idf is not None:
      return self._idf
    elif (self.docs is not None) and (self.terms is not None):
      self._compute_tf_idf()
      return self._idf
    else:
      raise RuntimeError("Either documents or pre-computed tf-idf should be given")

  @idf.setter
  def idf(self, value):
    assert isinstance(value, dict)
    self._idf = value

  def _compute_tf_idf(self):
    assert self.docs is not None
    self._tf, self._idf = utils.get_tf_idf(self.docs, self.terms, self.doc_prior)

  @property
  def top_terms(self):
    terms = utils.take_topk(self.term_scores, 20)
    ret = []
    for term in terms:
      if term.startswith("<phrase>"):
        ret.append(utils.strip_phrase_tags(term))
      else:
        ret.append(term)
    return ret

  def to_obj(self):
    # keep only necessary information
    node = dict()
    node["prefix"] = self.prefix
    node["term_scores"] = self.term_scores
    if self.doc_prior is not None:
      node["doc_prior"] = self.doc_prior
    else:
      node["doc_prior"] = {paper_id: 1 for paper_id in self.docs.keys()}
    if self.term_prior is not None:
      node["term_prior"] = self.term_prior
    else:
      node["term_prior"] = {term: 1 for term in self.terms}
    return node

  def save_readable(self, output_dir):
    output_dir = plib.Path(output_dir)
    assert output_dir.is_dir()
    with plib.Path(output_dir, "terms.txt").open("w") as f:
      for term, weight in self.term_scores.items():
        f.write(f"{term}\t{weight}\n")
    with plib.Path(output_dir, "documents.txt").open("w") as f:
      for doc, prior in self.doc_prior.items():
        f.write(f"{doc}\t{prior}\n")


class Taxonomy:

  def __init__(
      self,
      D,
      T,
      G: HIN,
      background_documents=None,
      background_tf=None,
      background_idf=None,
  ):
    self.G = G
    self.D = D
    self.T = T
    # Embedding by levels
    self._level_embed: List[LevelEmbed] = list()
    if background_idf and background_tf:
      self.background_node = TaxoNode("-", {"": []}, [], tf=background_tf, idf=background_idf)
    self._init_root()

  def _init_root(self):
    doc_prior = {paper_id: 1 for paper_id in self.D}
    term_prior = {term: 1 for term in self.T}
    self.root = TaxoNode("0", self.D, self.T, doc_prior, term_prior)

  def add_node(self, prefix, node):
    prefix = parse_prefix(prefix)
    if len(prefix) == 1:
      if self.background_node:
        self.background_node.children.append(node)
        node.parent = self.background_node
      self.root = node
    else:
      parent = self.root
      for level, order in enumerate(prefix):
        if level == 0:
          continue
        while len(parent.children) < order + 1:
          # create placeholder node
          new_prefix = "/".join(map(str, prefix[:level] + [len(parent.children)]))
          new_node = TaxoNode(new_prefix, {}, [])
          parent.add_child(new_node)
        if level == len(prefix) - 1:  # insert the node
          prev_node = parent.children[order]
          node.children = prev_node.children
          node.parent = parent
          parent.children[order] = node
        else:  # walk to lower level
          parent = parent.children[order]

  def siblings(self, node):
    if node == self.root:
      return []
    candidates = node.parent.children
    siblings = []
    for n in candidates:
      if node == n:
        continue
      siblings.append(n)
    return siblings

  def save(self, path):
    plib.Path(path).mkdir(parents=True, exist_ok=True)
    taxo_file = plib.Path(path, "taxodump.json.gz")
    # emb_file = plib.Path(path, "embedding.pkl")
    current_level = [self.root]
    next_level = []
    # traverse by level
    tx = dict()
    while True:
      for node in current_level:
        tx[node.prefix] = node.to_obj()
        next_level.extend(node.children)
      if not next_level:
        break
      current_level = next_level
      next_level = []
    with gzip.open(taxo_file, "wt", encoding="utf-8") as f:
      json.dump(tx, f)

  @staticmethod
  def load(path, corpus, T, G, background_tf=None, background_idf=None):
    taxo_file = plib.Path(path, "taxodump.json.gz")
    # emb_file = plib.Path(path, "embedding.pkl")
    taxo = Taxonomy(corpus, T, G, background_tf=background_tf, background_idf=background_idf)
    with gzip.open(taxo_file, "rt", encoding="utf-8") as f:
      tx = json.load(f)
    for prefix, data in tx.items():
      D = dict()
      for paper_id in data["doc_prior"].keys():
        D[paper_id] = corpus[paper_id]
      T = list(data["term_prior"].keys())
      node = TaxoNode(prefix, D, T, WD=data["doc_prior"], WT=data["term_prior"])
      logger.debug(f"load node {node.prefix}")
      # logger.debug(f"clustering probs is None? {node.clustering_probs is None}")
      taxo.add_node(prefix, node)

    return taxo

  def visualize(self, fname, shrink_root=True, max_level=4):
    graph = pydot.Dot(graph_type="digraph")
    current_level = [self.root]
    next_level = []
    level = 0
    while True:
      logger.debug(f"Level {level}")
      level += 1
      for node in current_level:
        if len(node.top_terms) > 0:
          # node.term_per_node = term_per_node
          if shrink_root and node.prefix == "0":
            node_root = pydot.Node(node.prefix, shape="record", label="{*}")
            graph.add_node(node_root)
          else:
            graph.add_node(node.dot_repr)
        logger.debug(f"{len(node.children)} children")
        for child in node.children:
          top_terms = child.top_terms
          logger.debug(f"{len(top_terms)} top terms")
          if len(child.top_terms) > 0:
            # child.term_per_node = term_per_node
            graph.add_node(child.dot_repr)
            if shrink_root and node.prefix == "0":
              graph.add_edge(pydot.Edge(node_root, child.dot_repr))
            else:
              graph.add_edge(pydot.Edge(node.dot_repr, child.dot_repr))
            next_level.append(child)
      if len(next_level) == 0 or level >= max_level:
        break
      current_level = next_level
      next_level = []
    graph.write(fname, format='pdf')

  def save_readable(self, output_dir):
    output_dir = plib.Path(output_dir)
    assert output_dir.is_dir()
    current_level = [self.root]
    next_level = []
    level = 0
    while True:
      level += 1
      for node in current_level:
        curr_dir = output_dir.joinpath(node.prefix)
        if not curr_dir.is_dir():
          curr_dir.mkdir(parents=False)
        node.save_readable(curr_dir)

        for child in node.children:
          if len(child.top_terms) > 0:
            next_level.append(child)
      if len(next_level) == 0:
        break
      current_level = next_level
      next_level = []
