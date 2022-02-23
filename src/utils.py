"""Utility functions for graph and text processing."""

import logging
import math
import os
import re
from collections import defaultdict
from operator import itemgetter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import Memory
from tqdm.auto import tqdm

memory = Memory("cache", verbose=1)

logger = logging.getLogger(__name__)


# helper functions for igraph and graph-tool
def sparse_to_edgelist(A):
  sources, targets = A.nonzero()
  edgelist = list(zip(sources, targets))
  return edgelist


def sparse_to_ig(A):
  import igraph as ig
  edgelist = sparse_to_edgelist(A)
  n_nodes = A.shape[0]
  G = ig.Graph(n=n_nodes, edges=edgelist)
  return G


def sparse_to_edges(A):
  COO = A.tocoo()
  return COO.row, COO.col, COO.data


def edges_to_sparse(row, col, data):
  return sp.csr_matrix((data, (row, col)))


def normalize(A):
  # normalize matrices
  rowsum = np.array(A.sum(1), dtype=np.float32)
  r_inv = np.power(rowsum, -0.5).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  columnsum = np.array(A.sum(0), dtype=np.float32)
  c_inv = np.power(columnsum, -0.5).flatten()
  c_inv[np.isinf(c_inv)] = 0.
  c_mat_inv = sp.diags(c_inv)
  A_norm = r_mat_inv * A * c_mat_inv
  return A_norm


def row_normalize(A):
  rowsum = np.array(A.sum(1), dtype=np.float32)
  r_inv = np.power(rowsum, -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  A_norm = r_mat_inv * A
  return A_norm


def load_documents(data_dir) -> Dict[str, List[str]]:
  """Load documents and find phrase in documents."""
  indices = []
  with open(os.path.join(data_dir, "indices.txt"), "r") as f:
    for line in f:
      indices.append(line.strip())

  phrases = []
  regex = re.compile(r"<phrase>(.+?)</phrase>")
  with open(os.path.join(data_dir, "text_documents.txt"), "r") as f:
    for line in f:
      ph = regex.findall(line)
      ph = ["<phrase>" + p.replace(" ", "_") + "</phrase>" for p in ph]
      phrases.append(ph)

  assert len(phrases) == len(indices)
  corpus = dict(zip(indices, phrases))
  return corpus


def load_graph(
    path: str = "../data/",
    force_undirected=False,
    remove_citation=False,
    remove_weights=False,
):
  """Load HIN from edge.txt and node_info.txt

  Returns:
    A: The adjacency matrix in whole, in scipy.sparse.csr_matrix format.
    node_info: A dictionary which maps node id to its type and more
    properties.
  """
  from hin import NodeInfo
  node_count = defaultdict(int)
  edge_count = 0
  node_info: Dict[int, NodeInfo] = dict()
  with open(os.path.join(path, "node_info.txt"), "r") as f:
    for line in f:
      tok = line.strip().split("\t")
      nid, t = int(tok[0]), tok[1]
      additional_info = tuple(tok[2:])
      node_info[nid] = NodeInfo(t, additional_info)
      node_count[t] += 1

  row = []
  column = []
  data = []
  with open(os.path.join(path, "edges.txt"), "r") as f:
    n_nodes, n_edges = list(map(int, next(f).strip().split()))
    for line in f:
      tok = line.split()
      if len(tok) == 2:  # no edge weight
        s, t = int(tok[0]), int(tok[1])
        w = 1
      elif len(tok) == 3:  # has edge weight
        s, t = int(tok[0]), int(tok[1])
        if not remove_weights:
          w = float(tok[2])
        else:
          w = 1
      if remove_citation and node_info[s].node_type == "P" \
        and node_info[t].node_type == "P" :
        continue
      edge_count += 1
      if force_undirected:
        row.extend([s, t])  # make it undirected!!
        column.extend([t, s])
        data.extend([w, w])
      else:
        row.extend([s])
        column.extend([t])
        data.extend([w])
  all_nodes = sum(node_count.values())
  node_count_str = ""
  for t, v in node_count.items():
    node_count_str += f"#{t}: {v}  "
  logger.info(f"#nodes {all_nodes}")
  logger.info(node_count_str)
  logger.info(f"#edges {edge_count}")

  A = sp.csr_matrix((data, (row, column)), shape=(n_nodes, n_nodes))

  return A, node_info


def add_phrase_tags(phrase):
  # data mining -> <phrase>data_mining</phrase>
  return "<phrase>" + phrase.lower().replace(" ", "_") + "</phrase>"


def strip_phrase_tags(phrase):
  # <phrase>data_mining</phrase> -> data mining
  return phrase[8:-9].replace("_", " ")


def take_topk(dictionary, K, return_tuple=False, return_dict=False):
  d = sorted(dictionary.items(), key=itemgetter(1), reverse=True)
  if return_tuple:
    return d[:K]
  if return_dict:
    return {k: v for (k, v) in d[:K]}
  else:
    return [item[0] for item in d][:K]


def get_tf_idf(
    documents: Dict[str, List[str]],
    phrases: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    progress_bar=False,
) -> Tuple[Dict[str, float], Dict[str, float]]:
  """Compute tf-idf.

  Documents are weighted.

  Args:
      documents: dict of documents, each value is in list of terms format.
      phrases: a list of phrases to compute tf-idf
      weights: weight on documents

  Returns:
      tf: dict of {phrase: tf}
      idf: dict of {phrase: idf}
  """
  if phrases is None:
    phrases_set: Set[str] = set()
    for doc in documents.values():
      phrases_set |= set(doc)
    phrases = list(phrases_set)

  if weights is None:
    weights = {paper_id: 1 for paper_id in documents.keys()}
  else:
    assert len(weights) == len(documents)
  W = np.zeros(len(documents))

  phrase2idx = dict()
  nu = 0
  for phrase in phrases:
    phrase2idx[phrase] = nu
    nu += 1

  # count matrix
  rows = []
  cols = []
  vals = []
  n_all = 0.
  i = 0
  for paper_id, doc in tqdm(documents.items(), disable=(not progress_bar)):
    n_all += len(doc) * weights[paper_id]
    W[i] = weights[paper_id]
    for phrase in doc:
      try:
        j = phrase2idx[phrase]
      except KeyError as e:
        continue
      rows.append(i)
      cols.append(j)
      vals.append(1)
    i += 1

  count_mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(documents), len(phrases)))

  # get tf & idf
  tf_vals = count_mat.T @ W / n_all
  df_vals = count_mat.minimum(1).T @ W

  tf = dict()
  idf = dict()
  W_sum = W.sum()
  for phrase, idx in phrase2idx.items():
    tf[phrase] = tf_vals[idx]
    idf[phrase] = math.log(W_sum / df_vals[idx]) if df_vals[idx] > 0 else 0
  return tf, idf


@memory.cache
def get_tf_idf_from_file(fname, phrases):
  phrase2idx = dict()
  nu = 0
  for phrase in phrases:
    phrase2idx[phrase] = nu
    nu += 1

  rows = []
  cols = []
  vals = []
  n_all = 0
  i = 0
  regex = re.compile(r"<phrase>(.+?)</phrase>")
  with open(fname, "r") as f:
    for line in tqdm(f):
      ph = regex.findall(line)
      ph = ["<phrase>" + p.replace(" ", "_") + "</phrase>" for p in ph]
      n_all += len(ph)
      for phrase in ph:
        try:
          j = phrase2idx[phrase]
        except KeyError as e:
          continue
        rows.append(i)
        cols.append(j)
        vals.append(1)
      i += 1

  count_mat = sp.csr_matrix((vals, (rows, cols)), shape=(i, len(phrases)))
  tf_vals = (np.array(count_mat.sum(0)).flatten().astype(np.float64) + 1) / n_all
  df_vals = np.array(count_mat.minimum(1).sum(0)).flatten().astype(np.float64)

  tf = dict()
  idf = dict()
  n_docs = i
  for phrase, idx in phrase2idx.items():
    tf[phrase] = tf_vals[idx]
    idf[phrase] = math.log(n_docs / df_vals[idx]) if df_vals[idx] > 0 else 0
  return tf, idf


def ensure_dir(path):
  p = Path(path)
  if not p.is_dir():
    p.mkdir(parents=True)
