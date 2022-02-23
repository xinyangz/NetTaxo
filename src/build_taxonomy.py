"""NetTaxo main file"""

import logging
import logging.config
import pathlib as plib
import pickle as pkl
from typing import Dict, List

import numpy as np

import clustering as clus
import config
import contrast_analysis as contrast
import local_embedding as loc_emb
import utils
from config import handle_args
from hin import HIN
from motif_embed import (generate_motif_context, motif_selection_sampling, sample_motif_context)
from motif_embed import MotifMatcher, Motif_KPA, Motif_KPV, Motif_KP, Motif_KPVY, Motif_KPAA
from taxonomy import TaxoNode, Taxonomy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _term2nid(G: HIN, terms: List[str]) -> List[int]:
  return G.find_by_entity_ids("K", terms)


class NetTaxo:

  def __init__(self, motif_matchers: List[MotifMatcher], tf_lift=1.5, idf_lift=1.5, damping=0.8, conf_motif=None):
    self.motif_matchers = motif_matchers
    self.tf_lift = tf_lift
    self.idf_lift = idf_lift
    self.damping = damping
    # use can specify a confident motif pattern that helps embedding and clustering
    self.conf_motif = conf_motif

  def set_background(self, tf: Dict[str, float], idf: Dict[str, float]):
    """Set background corpus for contrastive analysis."""
    # we don't really care about documents and terms
    self.background_node = TaxoNode("-", None, None, tf=tf, idf=idf)

  def build(self, taxo: Taxonomy, levels: List[int]):
    """Build taxonomy.

    Args:
      taxo: A mutable Taxonomy object to store the built taxonomy.
      levels: A list indicating the output taxonomy architecture, i.e. how many
        nodes on each level, e.g. [5, 4, 3].
      max_level: Maximum number of taxonomy to be built.
    """
    logger.info("Start building taxonomy")
    # first we set the auxilary background node
    taxo.root.set_parent(self.background_node)

    level = 0
    curr_level_nodes = [taxo.root]
    next_level_nodes: List[TaxoNode] = []
    # we need one more run to compute term score for all leaf nodes
    # this auxiliary level will stop before step 2 local embedding
    max_level = len(levels)
    levels.append(0)
    for level in range(len(levels)):
      logger.info(f"Processing level {level}")
      for node in curr_level_nodes:
        self.expand_node(taxo, node, level, levels[level], max_level=max_level)
        next_level_nodes.extend(node.children)
      curr_level_nodes = next_level_nodes
      next_level_nodes = []
    # unset background node
    taxo.root.set_parent(None)

  def expand_node(self, taxo: Taxonomy, node: TaxoNode, level: int, n_children: int, max_level: int):
    """Expand a taxonomy node.

    Args:
      taxo: The Taxonomy object.
      node: The node to expand.
      level: Current level of taxonomy.
      n_children: Number of children to expand to next level.
    """
    logger.info(f"Expand node {node.prefix}")
    logger.info("1. Contrast analysis for term scoring")
    if node.parent is None:
      raise RuntimeError("Contrastive analysis assumes parent node is set.")
    else:
      term_scores = contrast.node_contrast_analysis(
          node,
          node.parent,
          taxo.siblings(node),
          self.tf_lift * (config.LEVEL_DECAY**level),
          self.idf_lift * (config.LEVEL_DECAY**level),
      )
    for term, old_score in term_scores.items():
      term_scores[term] = old_score * node.term_prior[term]
    node.term_scores = term_scores

    logger.debug("Top terms for this node")
    logger.debug(str([utils.strip_phrase_tags(phrase) for phrase in utils.take_topk(term_scores, 20)]))

    logger.info(f"check stopping criteria, level {level} >= {max_level} is {level >= max_level}")
    if level >= max_level:
      return

    logger.info("Generate motif context")
    generate_motif_context(args.data_dir, level, taxo.G, node.terms, node.docs, self.motif_matchers)
    sample_motif_context(args, level, self.conf_motif)

    logger.info("2. Local embedding")
    word_embed, net_embed = loc_emb.local_embedding(node, args.data_dir)
    wv_word = word_embed.syn0
    wv_net = net_embed.syn0

    logger.info("3. Term clustering")
    logger.debug(f"#term_scores {len(term_scores)}")
    topk = min(config.N_TOP_TERMS, int(config.TOP_TERMS_PCT * len(term_scores)))
    topk_terms = utils.take_topk(term_scores, topk)
    clus_labels, aligned_terms = clus.term_clustering(topk_terms, wv_word, n_clusters=n_children)
    clus_labels_net, aligned_terms_net = clus.term_clustering(topk_terms, wv_net, n_clusters=n_children)
    map_ab = clus.align_clustering(clus_labels, clus_labels_net)

    logger.info("4. Anchor phrase selection")
    # anchor phrase selection w/ intersection
    # WT_clusters = []  # term weights
    term_weights_clusters = []
    for i in range(n_children):
      # get all terms in the cluster
      clus_terms_word = set(clus.get_cluster_terms(clus_labels, aligned_terms, i))
      clus_terms_net = set(clus.get_cluster_terms(clus_labels_net, aligned_terms_net, map_ab[i]))
      clus_terms = clus_terms_word & clus_terms_net
      term_nids = _term2nid(taxo.G, clus_terms)
      # get associated documents
      D_c, weights_c = clus.get_cluster_documents(taxo.G, node.docs, term_nids)
      # run contrastive analysis
      tf_c, idf_c = utils.get_tf_idf(D_c, clus_terms, weights=weights_c)
      next_level = level + 1
      term_scores_c = contrast.contrast_analysis(tf_c, idf_c, node.tf, node.idf,
                                                 self.tf_lift * (config.LEVEL_DECAY**next_level),
                                                 self.idf_lift * (config.LEVEL_DECAY**next_level))
      term_weights_clusters.append(term_scores_c)
      logger.debug("Cluster {}:: ".format(i) +
                   str([utils.strip_phrase_tags(phrase) for phrase in utils.take_topk(term_scores_c, 30)]))

    logger.info("5. Motif selection")
    cluster_seeds = []
    n_seed = config.N_ANCHOR_TERMS
    for i in range(n_children):
      seed_phrases = utils.take_topk(term_weights_clusters[i], n_seed)
      cluster_seeds.append(seed_phrases)

    motif_selection_sampling(args, level, cluster_seeds, keep_ratio=config.TOP_MOTIF_PCT)

    logger.info("6. Recompute embedding")
    joint_embed = loc_emb.joint_local_embedding(node, args.data_dir)
    wv_all = joint_embed.syn0

    logger.info("7. Soft clustering")
    clus_labels, aligned_terms, vmf = clus.soft_clustering(topk_terms, wv_all, n_clusters=n_children)

    logger.info("8. Generate next level")
    term_prior_clusters = []
    cluster_centers = []
    for i in range(n_children):
      # get all terms in the cluster
      clus_terms = clus.get_cluster_terms(clus_labels, aligned_terms, i)
      term_nids = _term2nid(taxo.G, clus_terms)
      # get associated documents
      D_c, weights_c = clus.get_cluster_documents(taxo.G, node.docs, term_nids)
      # run contrastive analysis
      tf_c, idf_c = utils.get_tf_idf(D_c, clus_terms, weights=weights_c)
      term_scores_c = contrast.contrast_analysis(tf_c, idf_c, node.tf, node.idf)
      term_prior_clusters.append(term_scores_c)
      logger.debug("Cluster {}:: ".format(i) +
                   str([utils.strip_phrase_tags(phrase) for phrase in utils.take_topk(term_scores_c, 30)]))

    # generate next level terms and documents
    # compute clustering probability
    X = []
    X_terms = []
    for term in node.terms:
      try:
        X.append(wv_all[term])
        X_terms.append(term)
      except:
        pass
    X = np.vstack(X)
    clustering_probs = clus.get_soft_cluster_probs(X, vmf.cluster_centers_, vmf.weights_, vmf.concentrations_)
    clustering_probs = clustering_probs.T
    for idx_c in range(n_children):
      # find words in each cluster
      terms_c = []
      term_prior_c = dict()
      for i in range(X.shape[0]):
        if clustering_probs[i, idx_c] > 2 * (1 / n_children):
          terms_c.append(X_terms[i])
          term_prior_c[X_terms[i]] = clustering_probs[i, idx_c]

      # find documents associated with each cluster
      ranking, clustering_probs_net = clus.populate_clustering(taxo.G, n_children, term_prior_clusters, damping=0.8)
      doc_prior_c = dict()
      docs_c = dict()
      for paper_id, paper_content in node.docs.items():
        nid = taxo.G.find_by_entity_id("P", paper_id)
        score = clustering_probs_net[nid, idx_c]
        if score <= 2 * (1 / n_children):
          continue
        docs_c[paper_id] = paper_content
        doc_prior_c[paper_id] = score

      node_c = TaxoNode(node.prefix + "/{}".format(idx_c), docs_c, terms_c, doc_prior_c, term_prior_c)
      curr = node_c
      node_c.set_parent(node)
      node.add_child(node_c)


def main():
  logger.warning("Start building taxonomy")
  # Load input: this includes reading network, text, and
  # a background corpus for contrastive analysis
  logger.info("Loading graph from file")
  A, node_info = utils.load_graph(args.data_dir, remove_citation=True, force_undirected=True)
  logger.info("Create HIN")
  G = HIN(A, node_info)

  logger.info("Load text")
  corpus = utils.load_documents(args.data_dir)

  motif_matchers = [Motif_KPV(), Motif_KPA(), Motif_KP(), Motif_KPVY(), Motif_KPAA()]

  intermediate_dir = plib.Path(args.data_dir, "intermediate")
  if not intermediate_dir.is_dir():
    logger.warning(f"Creating intermediate dir {intermediate_dir}")
    intermediate_dir.mkdir(parents=False)

  # we collect all phrases
  T = []  # terms / phrases
  for info in node_info.values():
    if info.node_type == "K":
      T.append(info.entity_id)

  D = corpus
  tf_bg, idf_bg = utils.get_tf_idf_from_file(plib.Path(args.data_dir, "background_documents.txt"), T)

  taxo = Taxonomy(D, T, G)

  builder = NetTaxo(motif_matchers,
                    tf_lift=args.tf_lift,
                    idf_lift=args.idf_lift,
                    damping=args.damping,
                    conf_motif=Motif_KPA().motif_name)

  # set background corpus for contrastive analysis
  builder.set_background(tf_bg, idf_bg)
  builder.build(taxo, args.levels)

  # save
  output_dir = plib.Path(args.output_dir, config.unique_id)
  if not output_dir.is_dir():
    output_dir.mkdir(parents=True)
  logger.info(f"Saving to {output_dir}")
  taxo.save(output_dir)

  logger.info("Saving complete")

  # generate output
  taxo.visualize(plib.Path(output_dir, f"vis.pdf"))
  taxo.save_readable(output_dir)


def cleanup():
  logger.info("Cleaning up")
  corpus = plib.Path(args.data_dir, "intermediate/corpus_{}.txt".format(config.unique_id))
  if corpus.is_file():
    corpus.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob("text_documents_*_{}.txt".format(config.unique_id)):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"local_*_{config.unique_id}.txt"):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"word_emb*_{config.unique_id}*.txt"):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"net_emb*_{config.unique_id}*.txt"):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"joint_emb*_{config.unique_id}*.txt"):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"net_context_*{config.unique_id}*.txt"):
    p.unlink()
  for p in plib.Path(args.data_dir, "intermediate").glob(f"g_context_*_{config.unique_id}*"):
    p.unlink()


if __name__ == "__main__":
  try:
    global args
    args = handle_args()
    utils.ensure_dir(args.log_dir)
    config.unique_id = f"{args.base_name}_" + config.unique_id
    dataset_name = args.data_dir.stem
    logging.config.fileConfig(
        "logging.conf",
        defaults={'logfilename': plib.Path(args.log_dir, f"{dataset_name}_{config.unique_id}.log")},
        disable_existing_loggers=False)
    logger.info(f"dataset name {dataset_name}")
    main()
  finally:
    cleanup()
    logger.info("Bye")
