import collections
import itertools
import logging
import pickle as pkl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from scipy.stats import entropy
from tqdm import tqdm

import config
from hin import HIN, _coo_to_nb_edgelist

logger = logging.getLogger(__name__)


class MotifMatcher(ABC):

  @property
  @abstractmethod
  def motif_name(self):
    pass

  @abstractmethod
  def match(self, G: HIN, document_ids: List[str], terms: List[str]) -> Tuple[sp.csr_matrix, List[str], List[str]]:
    pass


class Motif_KPV(MotifMatcher):

  @property
  def motif_name(self):
    return "KPV"

  def match(self, G, document_ids, terms):
    logger.info("Match KPV")
    row_labels = []
    col_labels = []
    P_indices = []
    K_indices = []
    for paper_id in document_ids:
      P_indices.append(G.find_by_entity_id("P", paper_id))

    for term in terms:
      row_labels.append(term)
      K_indices.append(G.find_by_entity_id("K", term))

    V_indices = G.type_idx["V"]
    for v in V_indices:
      col_labels.append(G.get_info(v))

    KP = G.A[K_indices, :]
    KP = KP[:, P_indices]
    PV = G.A[P_indices, :]
    PV = PV[:, V_indices]
    KPV = KP @ PV
    return KPV, row_labels, col_labels


class Motif_KPA(MotifMatcher):

  @property
  def motif_name(self):
    return "KPA"

  def match(self, G, document_ids, terms):
    logger.info(f"Match {self.motif_name}")
    row_labels = []
    col_labels = []
    P_indices = []
    K_indices = []
    for paper_id in document_ids:
      P_indices.append(G.find_by_entity_id("P", paper_id))

    for term in terms:
      row_labels.append(term)
      K_indices.append(G.find_by_entity_id("K", term))

    A_indices = G.type_idx["A"]
    for a in A_indices:
      col_labels.append(G.find_by_entity_id("A", a))
    KP = G.A[K_indices, :]
    KP = KP[:, P_indices]
    PA = G.A[P_indices, :]
    PA = PA[:, A_indices]
    KPA = KP @ PA
    return KPA, row_labels, col_labels


class Motif_KP(MotifMatcher):

  @property
  def motif_name(self):
    return "KP"

  def match(self, G, document_ids, terms):
    logger.info(f"Match {self.motif_name}")
    row_labels = []
    col_labels = []
    P_indices = []
    K_indices = []
    for paper_id in document_ids:
      col_labels.append(paper_id)
      P_indices.append(G.find_by_entity_id("P", paper_id))

    for term in terms:
      row_labels.append(term)
      K_indices.append(G.find_by_entity_id("K", term))

    KP = G.A[K_indices, :]
    KP = KP[:, P_indices]
    return KP, row_labels, col_labels


class Motif_KPVY(MotifMatcher):
  """Paper + Year range motif"""

  @property
  def motif_name(self):
    return "KPVY"

  def match(self, G: HIN, document_ids, terms):
    logger.info(f"Match {self.motif_name}")
    row_labels = []
    P_indices = []
    K_indices = []
    for paper_id in document_ids:
      P_indices.append(G.find_by_entity_id("P", paper_id))

    for term in terms:
      row_labels.append(term)
      K_indices.append(G.find_by_entity_id("K", term))

    KP = G.A[K_indices, :]
    KP = KP[:, P_indices]

    Y_split = []
    for i in range(1965, 2021, 5):
      Y_split.append(i)

    col_labels = []
    V_indices = G.type_idx["V"]
    n_venues = len(V_indices)
    for y in Y_split:
      for v in V_indices:
        col_labels.append(G.get_info(v) + " " + str(y))

    Y_indices = G.type_idx["Y"]
    PY = G.A[P_indices, :]
    PY = PY[:, Y_indices]
    PV = G.A[P_indices, :]
    PV = PV[:, V_indices]
    KPVY = sp.dok_matrix((KP.shape[0], n_venues * len(Y_split)))
    for i in tqdm(range(KP.shape[0])):
      for p_idx in KP[i, :].nonzero()[1]:
        try:
          v_idx = PV[p_idx].nonzero()[1][0]
          y_idx = PY[p_idx].nonzero()[1][0]
          year = int(G.get_info(y_idx))
        except:
          continue
        for k in range(len(Y_split)):
          if Y_split[k] > year:
            offset = k
            break
        KPVY[i, v_idx + offset * n_venues] += 1

    KPVY = KPVY.tocsr()
    return KPVY, row_labels, col_labels


class Motif_KPAA(MotifMatcher):

  def __init__(self, freq_threshold=5):
    self.freq_threshold = freq_threshold

  @property
  def motif_name(self):
    return "KPAA"

  def match(self, G: HIN, document_ids, terms):
    logger.info(f"Match {self.motif_name}")
    row_labels = []
    P_indices = []
    K_indices = []
    for paper_id in document_ids:
      P_indices.append(G.find_by_entity_id("P", paper_id))

    keyword2idx = dict()
    for i, term in enumerate(terms):
      row_labels.append(term)
      keyword2idx[term] = i
      K_indices.append(G.find_by_entity_id("K", term))

    KP = G.A[K_indices, :]
    KP = KP[:, P_indices]

    col_labels = []
    author_pairs = list()
    # scan through all papers and get all author pairs
    for nid in tqdm(G.type_idx["P"]):
      neighbors = G.neighbors(nid)
      authors = []
      for i in neighbors:
        if G.get_type(i) == "A":
          authors.append(i)
      for a1, a2 in itertools.combinations(authors, r=2):
        author_pairs.append(str(a1) + "-" + str(a2))

    # select pairs with more than freq_threshold occurrence
    c = collections.Counter(author_pairs)
    freq_author_pairs = set([pair for pair, cnt in c.items() if cnt >= self.freq_threshold])
    logger.debug(f"frequent author pairs {len(freq_author_pairs)}")

    # scan through all papers and build bipartite graph
    author2idx = {}  # author-pair -> bipartite graph idx
    for i, author_pair in enumerate(freq_author_pairs):
      author2idx[author_pair] = i
      a1, a2 = author_pair.split("-")
      a1, a2 = int(a1), int(a2)
      a1_name, a2_name = G.get_info(a1), G.get_info(a2)
      col_labels.append(a1_name + " - " + a2_name)

    n_keyword = len(terms)
    n_author = len(author2idx)

    keyword_set = set(terms)

    KPAA = sp.dok_matrix((n_keyword, n_author))

    for nid in tqdm(P_indices):
      neighbors = G.neighbors(nid)
      authors = []
      keywords = []
      for i in neighbors:
        if G.get_type(i) == "A":
          authors.append(i)
        elif G.get_type(i) == "K":
          keywords.append(G.get_info(i))
        pairs = []
        for a1, a2 in itertools.combinations(authors, r=2):
          if str(a1) + "-" + str(a2) in freq_author_pairs:
            pairs.append(str(a1) + "-" + str(a2))
        for kwd in keywords:
          if kwd not in keyword_set:
            continue
          row_id = keyword2idx[kwd]
          for pair in pairs:
            col_id = author2idx[pair]
            KPAA[row_id, col_id] += 1

    KPAA = KPAA.tocsr()
    return KPAA, row_labels, col_labels


def generate_motif_context(data_dir, level, G: HIN, terms: List[str], documents: Dict[str, List[str]],
                           motif_matchers: List[MotifMatcher]):
  """Generate term-motif bipartite graph"""
  logger.info("generate motif context")

  if level == 0:
    # level 0 motif matching don't have to be rerun
    unique_id = "lv0"
    all_done = True
    for motif_matcher in motif_matchers:
      motif_name = motif_matcher.motif_name
      if not Path(data_dir, f"intermediate/g_context_{motif_name}_{unique_id}.npz").is_file():
        all_done = False
        break
    logger.debug(f"First level motif context all done {all_done}")
    if all_done:
      return
  else:
    unique_id = config.unique_id

  document_ids = list(documents.keys())
  for matcher in motif_matchers:
    motif_name = matcher.motif_name
    context_mat, row_labels, col_labels = matcher.match(G, document_ids, terms)
    sp.save_npz(Path(data_dir, f"intermediate/g_context_{motif_name}_{unique_id}.npz"), context_mat)
    with Path(data_dir, f"intermediate/g_context_{motif_name}_{unique_id}_labels.pkl").open("wb") as f:
      pkl.dump({"row_labels": row_labels, "col_labels": col_labels}, f)


def authority_ranking(W, init_seeds_a=[], init_seeds_b=[], rank_iter=4, verbose=True):

  def _normalize(A):
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

  W_norm = _normalize(W)
  # initialization
  if init_seeds_a is not None and len(init_seeds_a) > 0:
    assert (init_seeds_b is None or len(init_seeds_b) == 0), "Seeds can only be provided for one type"
    vec_A = np.zeros(W.shape[0])
    vec_A[init_seeds_a] = 1 / len(init_seeds_a)
  else:
    vec_A = np.ones(W.shape[0]) / W.shape[0]
  if init_seeds_b is not None and len(init_seeds_b) > 0:
    assert (init_seeds_a is None or len(init_seeds_a) == 0), "Seeds can only be provided for one type"
    vec_B = np.zeros(W.shape[1])
    vec_B[init_seeds_b] = 1 / len(init_seeds_b)
  else:
    vec_B = np.ones(W.shape[1]) / W.shape[1]
  # propagate
  n_iter = 0
  while n_iter < rank_iter:
    if init_seeds_b is not None and len(init_seeds_b) > 0:
      vec_A_ = W_norm * vec_B
      vec_B = W_norm.T * vec_A_
    else:
      vec_B = W_norm.T * vec_A
      vec_A_ = W_norm * vec_B
    if np.linalg.norm(vec_A_ - vec_A) < 1e-3:
      if verbose:
        print("Ranking converge at iter", n_iter)
      break
    vec_A = vec_A_
    n_iter += 1
  vec_A = vec_A / vec_A.sum()
  vec_B = vec_B / vec_B.sum()
  return vec_A, vec_B


def clustering_propagation(cluster_seeds, KC, max_iter=4):
  n_clusters = len(cluster_seeds)
  scores = np.zeros((KC.shape[1], n_clusters))
  for i in range(n_clusters):
    _, scores[:, i] = authority_ranking(KC, init_seeds_a=cluster_seeds[i], rank_iter=max_iter)
  return scores


def context_node_score(cluster_seeds, context_mat, label_dict=None):
  n_clusters = len(cluster_seeds)
  scores = clustering_propagation(cluster_seeds, context_mat)
  scores_one = clustering_propagation(cluster_seeds, context_mat, max_iter=1)
  node_score = dict()
  context_mat = context_mat.tocsc()
  for i in range(scores.shape[0]):
    if scores_one[i].sum() == 0:
      node_entropy = 0
    else:
      node_entropy = (1 - entropy(scores_one[i]) / np.log(scores_one[i].shape[0]))
    node_score[i] = np.mean(scores[i]) * node_entropy
  if label_dict is not None:
    labeled_score = dict()
    for k, v in node_score.items():
      labeled_score[label_dict[k]] = v
    node_score = labeled_score
  return node_score


def context_node_selection(context_mat, node_scores, threshold):
  kept_nodes = []
  for node, score in node_scores.items():
    if score > threshold:
      kept_nodes.append(node)
  masked_context_mat = context_mat[:, kept_nodes]
  return masked_context_mat


def sample_context_mat(context_mat, inits, sample_length):
  # warning: context_mat is a dense matrix and row normalized
  seqs = np.zeros((inits.shape[0], sample_length), dtype=np.int32)
  for i in range(inits.shape[0]):
    probs = context_mat[inits[i]]
    try:
      seqs[i] = np.random.choice(context_mat.shape[1], p=probs, size=(sample_length,))
    except:
      print(i)
      print(probs)
      break
  return inits, seqs


@njit(parallel=True, cache=True)
def _uniform_sample_context_mat(starts, edgelist_flat, edgelist_ranges, sample_length):
  seqs = np.ones((len(starts), sample_length), dtype=np.int32) * -1
  for i in prange(starts.shape[0]):
    curr = starts[i]
    start, end = edgelist_ranges[curr][0], edgelist_ranges[curr][1]
    if end - start + 1 <= 0:
      continue
    sample_indices = np.random.choice(end - start + 1, size=(sample_length,))
    for j in range(sample_length):
      seqs[i, j] = edgelist_flat[start + sample_indices[j]]
  return seqs


def uniform_sample_context_mat(context_mat, sample_per_node, sample_length):
  rows, cols = context_mat.nonzero()
  edgelist_flat, edgelist_ranges = _coo_to_nb_edgelist(rows, cols, context_mat.shape[0])
  inits = np.repeat(np.arange(context_mat.shape[0], dtype=np.int32), sample_per_node)
  seqs = _uniform_sample_context_mat(inits, edgelist_flat, edgelist_ranges, sample_length)
  return inits, seqs


def motif_selection_sampling(args, level, cluster_seeds, keep_ratio=0.2):
  if level == 0:
    unique_id = "lv0"
  else:
    unique_id = config.unique_id

  # load context matrix
  logger.info("Sample motif context")
  dump_seqs = []
  context_files = Path(args.data_dir, "intermediate").glob(f"g_context_*_{unique_id}.npz")

  all_scores = []

  for f in context_files:
    logger.debug(f"Loading {f}")
    KC = sp.load_npz(f)
    basename = f.stem
    label_file = basename + "_labels.pkl"
    with f.with_name(label_file).open("rb") as f_label:
      obj = pkl.load(f_label)
      row_labels = obj["row_labels"]
    phrase2idx = {row_labels[i]: i for i in range(len(row_labels))}
    cluster_seeds_idx = []
    for clus in cluster_seeds:
      cluster_seeds_idx.append([phrase2idx[phrase] for phrase in clus])
    node_scores = context_node_score(cluster_seeds_idx, KC)
    all_scores.extend(node_scores.values())

  all_scores = sorted(all_scores, reverse=True)
  score_threshold = all_scores[int(keep_ratio * len(all_scores))]

  context_files = Path(args.data_dir,"intermediate")\
    .glob(f"g_context_*_{unique_id}.npz")
  logger.debug(f"Start sampling from {context_files}")
  for i, f in enumerate(context_files):
    logger.debug(f"Sampling {f}")
    KC = sp.load_npz(f)
    basename = f.stem
    label_file = basename + "_labels.pkl"
    with f.with_name(label_file).open("rb") as f_label:
      obj = pkl.load(f_label)
      row_labels = obj["row_labels"]
    phrase2idx = {row_labels[i]: i for i in range(len(row_labels))}
    cluster_seeds_idx = []
    for clus in cluster_seeds:
      cluster_seeds_idx.append([phrase2idx[phrase] for phrase in clus])
    node_scores = context_node_score(cluster_seeds_idx, KC)
    kept_context_mat = context_node_selection(KC, node_scores, score_threshold)
    logger.debug(f"Selected nodes {kept_context_mat.shape[0]}")

    # sample
    context_mat = kept_context_mat.toarray()
    context_mat += 1e-6
    normalizer = context_mat.sum(1)
    context_mat /= normalizer[:, np.newaxis]
    inits = np.repeat(np.arange(context_mat.shape[0], dtype=np.int32), 1)
    inits, seqs = uniform_sample_context_mat(context_mat, 40, 4)

    # append sequences
    for idx in range(len(inits)):
      if seqs[idx, 0] < 0:
        continue
      word = row_labels[inits[idx]]
      dump_seqs.append(word + " " + " ".join([str(i) + "_" + str(node) for node in seqs[idx]]) + "\n")

  np.random.shuffle(dump_seqs)
  logger.info("Dump to file")
  logger.debug(f"#seq {len(dump_seqs)}")
  with open(Path(args.data_dir, f"intermediate/net_context_all_{unique_id}.txt"), "w") as f:
    for line in dump_seqs:
      f.write(line)


def sample_motif_context(args, level, motif_name, sample_per_node=40, sample_length=4):
  if level == 0:
    unique_id = "lv0"
    if Path(args.data_dir, "intermediate/net_emb_0_lv0.syn0.txt").is_file():
      return
  else:
    unique_id = config.unique_id

  # load context matrix
  logger.info("Sample motif context")
  dump_seqs = []
  f = Path(args.data_dir, f"intermediate/g_context_{motif_name}_{unique_id}.npz")

  logger.debug(f"Sampling {f}")
  KC = sp.load_npz(f)
  basename = f.stem
  label_file = basename + "_labels.pkl"
  with f.with_name(label_file).open("rb") as f_label:
    obj = pkl.load(f_label)
    row_labels = obj["row_labels"]

  # sample
  context_mat = KC.toarray()
  context_mat += 1e-6
  normalizer = context_mat.sum(1)
  context_mat /= normalizer[:, np.newaxis]
  inits = np.repeat(np.arange(context_mat.shape[0], dtype=np.int32), 1)
  inits, seqs = uniform_sample_context_mat(context_mat, sample_per_node, sample_length)

  # append sequences
  for idx in range(len(inits)):
    if seqs[idx, 0] < 0:
      continue
    word = row_labels[inits[idx]]
    dump_seqs.append(word + " " + " ".join([str(node) for node in seqs[idx]]) + "\n")

  np.random.shuffle(dump_seqs)
  logger.info("Dump to file")
  with open(Path(args.data_dir, f"intermediate/net_context_{unique_id}.txt"), "w") as f:
    for line in dump_seqs:
      f.write(line)
