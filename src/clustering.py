import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from spherecluster import SphericalKMeans, VonMisesFisherMixture
from spherecluster.von_mises_fisher_mixture import _expectation

import utils
from hin import HIN

logger = logging.getLogger(__name__)


def get_cluster_terms(clus_labels: List[int], aligned_terms: List[str], cluster_id) -> List[str]:
  """A utility function to return all words in a cluster given clustering

  output.

  Args:
    clus_labels: A list of cluster labels.
    aligned_terms: A list of terms, aligned with clus_labels.
    cluster_id: The cluster to return.

  Returns:
    terms: All terms in the selected cluster.
  """
  terms = []
  for j in range(len(clus_labels)):
    if clus_labels[j] == cluster_id:
      terms.append(aligned_terms[j])
  return terms


def get_cluster_documents(G: HIN, D: Dict[str, List[str]],
                          term_nids: List[int]) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
  """Get all documents associated a set of terms.

  The weight on each document is computed by aggregating
  edge weights to all relevant term node neighbors.

  Args:
    G: The HIN.
    D: The set of documents to consider.
    term_nids: The set of terms in node id.

  Returns:
    documents: The set of associated documents.
    weights: The weight on each document indicating how
    much they are relevant.
  """
  documents = dict()
  weights: Dict[str, float] = defaultdict(float)
  for nid in term_nids:
    neighbors = G.neighbors(nid)  # assume P-K links are bi-directional
    for neighbor in neighbors:
      # filter out non paper nodes
      if G.get_type(neighbor) != "P":
        continue
      # filter out docs out of parent cluster
      paper_id = G.get_doc_id(neighbor)
      if paper_id not in D:
        continue
      weights[paper_id] += G.A[nid, neighbor]

  weights = dict(weights)
  for paper_id in weights.keys():
    documents[paper_id] = D[paper_id]
  return documents, weights


def align_clustering(clus_labels_a, clus_labels_b):
  """Reads clustering labels from two different sources and align them with best effort.

    To align clusters, we start from the first cluster in source A, and computes
    Jaccard similarity to each cluster in source B. We pick the best alignment, then
    move on to next cluster.

    Args:
        clus_labels_a: Clustering labels from source A, shape (N,).
        clus_labels_b: Clustering labels from source B, shape (N,).

    Returns:
        map_ab: A dictionary mapping cluster ID from A to B."""

  def _label2sets(clus_labels):
    label_set = set(clus_labels)
    label2id = dict()
    for i, label in enumerate(label_set):
      label2id[label] = i
    clusters = []
    for i in range(len(label_set)):
      clusters.append(set())
    for i in range(len(clus_labels)):
      clusters[label2id[clus_labels[i]]].add(i)
    return clusters

  def _jaccard(set_a, set_b):
    return len(set_a & set_b) / (len(set_a | set_b))

  map_ab = dict()
  clusters_a = _label2sets(clus_labels_a)
  clusters_b = _label2sets(clus_labels_b)
  assert len(clusters_a) == len(clusters_b), "#clusters must equal for source A and B"
  aligned = set()
  for i in range(len(clusters_a)):
    sim = []
    for j in range(len(clusters_b)):
      if j in aligned:
        sim.append(-1)
        continue
      sim.append(_jaccard(clusters_a[i], clusters_b[j]))
    best_match = np.argmax(sim)
    map_ab[i] = best_match
    aligned.add(best_match)
  return map_ab


def term_clustering(terms: List[str], wv: Dict[str, np.ndarray], n_clusters: int) -> Tuple[List[int], List[str]]:
  """Use spherical k-means to cluster word vectors.

  Args:
    terms: A list of terms to cluster.
    wv: A dictionary of word to their vectors.
    n_clusters: Number of output clusters.

  Returns:
    labels: A list of clustering assignment for each word.
    terms: A list of words, aligned with labels.
  """
  X = []
  X_terms = []
  n_out_of_vocab = 0
  logger.debug(f"#wv {len(wv)}")
  logger.debug(terms[:20])
  for term in terms:
    try:
      phrase = term
      emb = wv[phrase]
      X.append(emb)
      X_terms.append(phrase)
    except KeyError as e:
      n_out_of_vocab += 1

  logger.warning(f"{n_out_of_vocab} / {len(terms)} words out of vocab")
  logger.info(f"Clustering {len(X)} words")
  clus = SphericalKMeans(n_clusters=n_clusters)
  clus.fit(X)
  logger.info(f"Clustering complete")
  return clus.labels_, X_terms


def soft_clustering(terms: List[str], wv: Dict[str, np.ndarray], n_clusters: int) -> Tuple[List[int], List[str]]:
  """Use spherical vmf to cluster word vectors"""
  X = []
  X_terms = []
  n_out_of_vocab = 0
  for term in terms:
    try:
      phrase = term
      emb = wv[phrase]
      X.append(emb)
      X_terms.append(phrase)
    except KeyError as e:
      n_out_of_vocab += 1

  logger.debug(f"{n_out_of_vocab} / {len(terms)} words out of vocab")
  logger.debug(f"Clustering {len(X)} words")
  vmf_soft = VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft')
  vmf_soft.fit(X)

  return vmf_soft.predict(X), X_terms, vmf_soft


def get_soft_cluster_probs(X, centers, weights, concentrations):
  import sklearn.preprocessing as prep
  X = prep.normalize(X)
  return _expectation(X, centers, weights, concentrations)


def populate_clustering(G: HIN,
                        n_clusters: int,
                        WT_clusters: List[Dict[str, float]],
                        damping=0.8) -> Tuple[np.ndarray, np.ndarray]:
  """Populate clustering results from terms to whole graph by random walk w/ restart.

  Args:
    G: The HIN.
    n_clusters: Number of clusters.
    WT_clusters: A list of initial weights of terms in each cluster. These
      weights will be populated to the whole graph.
    damping: The damping factor for random walk. Larger means more restart
      probability.

  Returns:
    ranking: The ranking distribution over ALL nodes. Shape (n_nodes,
    n_clusters).
    clustering_probs: The clustering distribution of all nodes. Shape (n_nodes,
    n_clusters).
  """
  clustering_probs = np.zeros((G.num_nodes(), n_clusters), dtype=np.float64)
  for k in range(n_clusters):
    # get initial distribution using T_score
    T_score = list(WT_clusters[k].items())  # P_Ti
    # T_score = take_topk(WT_clusters[k], 20, return_tuple=True)
    phrases, scores = list(zip(*T_score))
    z = sum(WT_clusters[k].values())  # normalizer
    dist = np.zeros((G.num_nodes(),), dtype=np.float64)
    aligned_nids = G.find_by_entity_ids("K", phrases)
    for i in range(len(scores)):
      dist[aligned_nids[i]] = scores[i] / z

    # use random walk to populate clustering probabilities
    pr = G.ppr(damping=damping, init_probs=dist)
    clustering_probs[:, k] = pr
  ranking = clustering_probs
  clustering_probs = utils.row_normalize(clustering_probs)
  return ranking, clustering_probs
