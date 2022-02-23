"""Provide data structures for heterogeneous information networks."""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp

from numba import njit, prange
from utils import normalize, sparse_to_ig


@njit(parallel=True, cache=True)
def _walk(starts, edgelist_flat, edgelist_ranges, walk_len, damping=0.8):
  seqs = np.ones((len(starts), walk_len), dtype=np.int32) * -1
  for i in prange(starts.shape[0]):
    curr = starts[i]
    for j in range(walk_len):
      start, end = edgelist_ranges[curr][0], edgelist_ranges[curr][1]
      n_neighbors = end - start + 1
      if n_neighbors == 0:
        break
      idx = random.randint(0, n_neighbors - 1)
      curr = edgelist_flat[start + idx]
      seqs[i, j] = curr
      if random.random() > damping:
        # sample start node according to probs
        curr = starts[i]
  return seqs


@njit(parallel=True, cache=True)
def _walk_typed(starts, edgelist_flat, edgelist_ranges, metapath, walk_len, reset_prob=0.2):
  """Random walk following metapath, without reset.

  Args:
    starts: An ndarray of start nodes.
    edgelist_flat: Edgelist flattened to 1D, neighbors sorted by type.
    edgelist_ranges: (n_nodes, 2 * n_types) ndarray, storing start and end indices
      pointing to edgelist_flat.
    metapath: The metapath to follow.
    walk_len: Maximum length of each random walk. Walk may terminate when there's no
      valid next step.
    reset_prob: Reset probability, ignored for now.
  """
  seqs = np.ones((len(starts), walk_len), dtype=np.int32) * -1
  path_len = len(metapath)
  for i in prange(starts.shape[0]):
    curr = starts[i]
    # metapath_idx = 1
    for j in range(walk_len):
      next_type = metapath[(j + 1) % path_len]
      # assumes next_type is an integer within [0, n_types - 1]
      start, end = edgelist_ranges[curr][2 * next_type], edgelist_ranges[curr][2 * next_type + 1]
      n_neighbors = end - start + 1
      if n_neighbors <= 0:
        break
      idx = random.randint(0, n_neighbors - 1)
      curr = edgelist_flat[start + idx]
      seqs[i, j] = curr
      # metapath_idx = (metapath_idx + 1) % path_len
      # if random.random() < reset_prob:
      #   # sample start node according to probs
      #   curr = starts[i]
      #   metapath_idx = 1
  return seqs


@njit(parallel=True, cache=True)
def _walk_weighted(starts, edgelist_flat, weights_flat, edgelist_ranges, walk_len, damping):
  seqs = np.ones((len(starts), walk_len), dtype=np.int32) * -1
  W_sum = np.ones((edgelist_ranges.shape[0],), dtype=np.float32)
  # computes sum of edge weights for all nodes
  for i in range(edgelist_ranges.shape[0]):
    s = 0
    for j in range(edgelist_ranges[i, 0], edgelist_ranges[i, 1] + 1):
      s += weights_flat[j]
    W_sum[i] = s
  # random walk w/ weight sampling
  for i in prange(starts.shape[0]):
    curr = starts[i]
    for j in range(walk_len):
      start, end = edgelist_ranges[curr][0], edgelist_ranges[curr][1]
      n_neighbors = end - start + 1
      if n_neighbors == 0:
        break
      rnd = random.random()
      s = 0
      for k in range(start, end + 1):
        s += weights_flat[k]
        if s / W_sum[curr] > rnd:
          curr = edgelist_flat[k]
          break
      idx = random.randint(0, n_neighbors - 1)
      curr = edgelist_flat[start + idx]
      seqs[i, j] = curr
      if random.random() > damping:
        # restart
        curr = starts[i]
  return seqs


@njit(parallel=True, cache=True)
def _walk_with_reset(starts, reset_probs, edgelist_flat, edgelist_ranges, walk_len, damping=0.8):
  seqs = np.ones((len(starts), walk_len), dtype=np.int32) * -1
  for i in prange(starts.shape[0]):
    curr = starts[i]
    for j in range(walk_len):
      start, end = edgelist_ranges[curr][0], edgelist_ranges[curr][1]
      n_neighbors = end - start + 1
      if n_neighbors == 0:
        break
      idx = random.randint(0, n_neighbors - 1)
      curr = edgelist_flat[start + idx]
      seqs[i, j] = curr
      if random.random() > damping:
        # sample start node according to probs
        rnd = random.random()
        s = 0
        for k in range(starts.shape[0]):
          s += reset_probs[k]
          if s > rnd:
            curr = starts[k]
            break
  return seqs


@njit(cache=True)
def _accumulate_probs(samples, ind_max):
  counts = np.zeros(ind_max, dtype=np.float64)
  for i in range(samples.shape[0]):
    if samples[i] > 0:
      counts[samples[i]] += 1
  counts = counts / np.sum(counts)
  return counts


@njit(cache=True)
def _coo_to_nb_edgelist(rows: np.ndarray, cols: np.ndarray, N: int) -> Tuple[np.ndarray, np.ndarray]:
  """Convert coordinate format matrix to numba flattened edgelist.

  Args:
    rows: row indices of nonzero elements in a sparse matrix
    cols: column indices of nonzero elements in a sparse matrix, aligned with
      rows.
    N: number of rows / nodes.

  Returns:
    edgelist_flat: flattened edge list.
    edgelist_ranges: range indices to flattened edge list.
  """
  edgelist_ranges = np.zeros((N, 2), dtype=np.int32)
  # If also need weights, consider converting
  # to coo format
  j = 0
  start = 0
  for i in range(N):
    while j < rows.shape[0] and rows[j] == i:
      j += 1
    end = j - 1
    edgelist_ranges[i, 0] = start
    edgelist_ranges[i, 1] = end
    start = j
  return cols, edgelist_ranges


@njit(cache=True)
def _coo_to_nb_edgelist_typed(rows: np.ndarray, cols: np.ndarray, types: np.ndarray, n_nodes: int,
                              n_types: int) -> Tuple[np.ndarray, np.ndarray]:
  """Convert coordinate format matrix to numba flattened edgelist.

  Args:
    rows: row indices of nonzero elements in a sparse matrix, in ascent order.
    cols: column indices of nonzero elements in a sparse matrix, aligned with
      rows.
    types: (n_nodes,) ndarray of node types.
    n_nodes: number of rows / nodes.
    n_types: number of all node types.

  Returns:
    edgelist_flat: flattened edge list.
    edgelist_ranges: range indices to flattened edge list.
  """
  for i in range(rows.shape[0] - 1):
    assert rows[i] <= rows[i + 1]
  edgelist_flat = np.zeros((len(rows)), dtype=np.int32)
  neighbor_types = np.zeros((len(rows)), dtype=np.int32)  # this is more than enough
  edgelist_ranges = np.zeros((n_nodes, 2 * n_types), dtype=np.int32)
  j = 0
  start = 0
  for i in range(n_nodes):
    while j < rows.shape[0] and rows[j] == i:
      j += 1
    end = j - 1
    n_neighbors = end - start + 1
    for k in range(n_types):
      edgelist_ranges[i, 2 * k] = start
      edgelist_ranges[i, 2 * k + 1] = start - 1
    if end < start:
      continue

    # sorts neighbors by type
    for k in range(start, end + 1):
      node = cols[k]
      neighbor_types[k - start] = types[node]
    sorted_index = np.argsort(neighbor_types[:n_neighbors])
    # copies sorted neighbors into edgelist
    for k in range(sorted_index.shape[0]):
      node = cols[start + sorted_index[k]]
      edgelist_flat[start + k] = node

    # since the neighbors are now sorted, the code scans
    # through all neighbors, and figure out start and end
    # of each node type
    curr_type = -1
    for k in range(sorted_index.shape[0]):
      node = cols[start + sorted_index[k]]
      t = types[node]
      if t != curr_type:
        edgelist_ranges[i, 2 * t] = start + k
        edgelist_ranges[i, 2 * curr_type + 1] = start + k - 1
        curr_type = t
    edgelist_ranges[i, 2 * curr_type + 1] = end
    start = j
  return edgelist_flat, edgelist_ranges


@dataclass(frozen=True)
class NodeInfo:
  """Class to keep track of associated type and attributes of a node in an HIN."""
  node_type: str
  additional: Tuple[str]  # additional info

  @property
  def entity_id(self):
    return self.additional[0]


class HIN:

  def __init__(self, A: sp.csr_matrix, node_info: Dict[int, NodeInfo]):
    self.A = A
    self.node_info = node_info
    self.node_types = self._init_node_types(node_info)
    self.G_ig = None
    self.nid2paper = None
    self.edgelist_flat = None
    self.edgelist_ranges = None
    self.entity_id2nid = dict()

    # maintain a node_type:[node_id] dictionary to support fast look up
    # of all nodes of a type
    self.type_idx = dict()
    for t in self.node_types:
      self.type_idx[t] = self._init_type_indices(t)

  def _init_node_types(self, node_info: List[NodeInfo]) -> List[str]:
    type_set = set()
    for info in node_info.values():
      type_set.add(info.node_type)
    return list(type_set)

  def _init_type_indices(self, type_name):
    inds = [i for i in range(len(self.node_info)) if self.node_info[i].node_type == type_name]
    return inds

  # ------ basic methods
  def num_nodes(self):
    return self.A.shape[0]

  def get_info(self, nid):
    return self.node_info[nid].entity_id

  def get_doc_id(self, nid):
    return self.get_info(nid)

  def get_type(self, uid):
    return self.node_info[uid].node_type

  def _build_entity2nid(self, entity_type):
    self.entity_id2nid[entity_type] = dict()
    for nid, info in self.node_info.items():
      if info.node_type == entity_type:
        self.entity_id2nid[entity_type][info.entity_id] = nid

  def find_by_entity_id(self, entity_type: str, entity_id: str) -> int:
    """Find node id by entity_id."""
    if entity_type not in self.entity_id2nid:
      self._build_entity2nid(entity_type)
    return self.entity_id2nid[entity_type].get(entity_id, -1)

  def find_by_entity_ids(self, entity_type: str, entity_ids: List[str]) -> List[int]:
    """Find multiple node id by entity_id."""
    if entity_type not in self.entity_id2nid:
      self._build_entity2nid(entity_type)
    indices: List[int] = []
    for entity_id in entity_ids:
      indices.append(self.find_by_entity_id(entity_type, entity_id))
    return indices

  def get_type_indices(self, type_name):
    """ Get all node indices of type given

            Input: given type
            Output: a list of indices
    """
    return self.type_idx[type_name]

  # ------ page rank methods
  def _ensure_ig(self):
    """ Helper function to create igraph instance if not exists"""
    if self.G_ig is None:
      self.G_ig = sparse_to_ig(self.A)

  def _ensure_numba(self):
    if self.edgelist_flat is None:
      rows, cols = self.A.nonzero()
      N = self.A.shape[0]
      self.edgelist_flat, self.edgelist_ranges = _coo_to_nb_edgelist(rows, cols, N)

  def ppr(self, seeds=None, damping=0.85, init_probs=None):
    """ Get personalized pagerank scores

            Input: a list of seed nodes to start, damping probability,
                   i.e. probability to restart
            Output: a list of PR scores for all nodes
    """
    self._ensure_ig()
    if seeds is not None:
      if not isinstance(seeds, list):
        seeds = [seeds]  # this is a dirty work around for single number input
      pr_vals = self.G_ig.personalized_pagerank(
          vertices=None,  # all
          reset_vertices=seeds,
          damping=damping)
      return pr_vals
    if init_probs is not None:
      pr_vals = self.G_ig.personalized_pagerank(
          vertices=None,  # all
          reset=init_probs,
          damping=damping)
      return pr_vals
    if seeds is None and init_probs is None:
      return self.pr(damping=damping)

  def ppr_mc(self, init: Dict[int, float], damping=0.85, n_walk=100, walk_len=100) -> List[float]:
    """Monte-Carlo Method for personalized pagerank.

    Args:
      init: A dictionary of all starting nodes, with their probabilities. The
        probabilities need not to be normalized.
      damping: Damping factor, 1 - damping is reset probability.

    Returns:
      pr_vals: A list of pagerank values for all nodes. List index is node
      index.
    """
    self._ensure_numba()
    # edgelist_flat, edgelist_ranges = self._numba_edge_repr()
    reset_nodes, reset_probs = list(zip(*init.items()))
    reset_nodes = np.array(reset_nodes, dtype=np.int32)
    reset_probs = np.array(reset_probs)
    assert np.sum(reset_probs) > 0
    reset_probs = reset_probs / np.sum(reset_probs)
    starts = np.random.choice(reset_nodes, size=((n_walk - 1) * reset_nodes.shape[0]), p=reset_probs)
    starts = np.hstack([reset_nodes, starts])
    walks = _walk_with_reset(starts, reset_probs, self.edgelist_flat, self.edgelist_ranges, walk_len)
    pr_vals = _accumulate_probs(walks.flatten(), self.num_nodes())
    return pr_vals.tolist()

  def pr(self, damping=0.85):
    self._ensure_ig()
    pr_vals = self.G_ig.pagerank(damping=damping, implementation="power", niter=10, eps=1e-4)
    return pr_vals

  def neighbors(self, nid):
    return self.A[nid].nonzero()[1]

  # ------ meta-path methods
  def _parse_path(self, path_str):
    path = []
    while True:
      r = path_str.find("->")
      l = path_str.find("<-")
      if r == -1 and l == -1:
        path.append((path_str, -1))
        break
      assert r != l
      if (r < l and r > 0) or l == -1:
        type_name = path_str[:r]
        path.append((type_name, 0))
        path_str = path_str[r + 2:]
      elif (r > l and l > 0) or r == -1:
        type_name = path_str[:l]
        path.append((type_name, 1))
        path_str = path_str[l + 2:]
    path_tuple = []
    for i in range(1, len(path)):
      path_tuple.append((path[i - 1][0], path[i][0], path[i - 1][1]))
    return path_tuple

  def type_matrix(self, source_type, target_type):
    M = self.A[self.get_type_indices(source_type), :]
    M = M[:, self.get_type_indices(target_type)]
    return M

  def path_matrix(self, path):
    # parse meta path
    path = self._parse_path(path)
    for idx, relation in enumerate(path):
      if idx == 0:
        M = self.type_matrix(relation[0], relation[1])
        if relation[2] == 1:
          M = M.T
      else:
        T = self.type_matrix(relation[0], relation[1])
        if relation[2] == 1:
          M = M * T.T
        else:
          M = M * T
    return M

  # ------ ranking methods
  def authority_ranking(self, relation):
    self.importance = np.ones(self.A.shape[0], dtype=np.float64)
    for indices in self.type_idx.values():
      self.importance[indices] /= len(indices)

    # pre-compute and normalize relation matrix
    W = self.path_matrix(relation)
    W = normalize(W)
    path = self._parse_path(relation)
    type_A = path[0][0]
    type_B = path[-1][1]
    # iterate till convergence
    n_iter = 0
    delta_ = 0.
    while n_iter < 1000:
      delta = 0.
      n_iter += 1
      # propagate importance
      indices_A = self.type_idx[type_A]
      indices_B = self.type_idx[type_B]
      new_importance = W.T * self.importance[indices_A]
      delta = np.linalg.norm(new_importance - self.importance[indices_B])
      self.importance[indices_B] = new_importance
      self.importance[indices_A] = W * self.importance[indices_B]

      delta_change = abs(delta - delta_)
      delta_ = delta
      if n_iter % 100 == 0:
        print("Iter", n_iter, "delta", delta, end="\r")
      if delta_change < 5e-5:
        print("Convergence at iter", n_iter)
        break

  # ------ slice subgraph methods
  def subgraph(self, subgraph_nodes):
    """ Slice a subgraph given a list of node indices """
    subgraph_nodes = sorted(subgraph_nodes)
    # slice adjacency matrix
    A = self.A[subgraph_nodes, :]
    A = A[:, subgraph_nodes]
    # slice node_info
    node_info = {}
    for i, uid in enumerate(subgraph_nodes):
      node_info[i] = self.node_info[uid]  # slice & remap indicies
    return HIN(A.copy(), node_info)
