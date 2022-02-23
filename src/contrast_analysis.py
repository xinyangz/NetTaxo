from typing import Dict, List

import numpy as np

from taxonomy import TaxoNode


def _contrast_eq(distinct: float, tf: float, idf: float, tf_lift: float = 1.5, idf_lift: float = 1.5) -> float:
  return distinct * (tf**tf_lift) * (idf**idf_lift)


def contrast_analysis(tf: Dict[str, float],
                      idf: Dict[str, float],
                      larger_tf: Dict[str, float],
                      larger_idf: Dict[str, float],
                      tf_lift=1.5,
                      idf_lift=1.5) -> Dict[str, float]:
  scores = dict()
  for phrase, freq in tf.items():
    larger_freq = larger_tf[phrase]
    idf_score = idf[phrase]
    scores[phrase] = (0 if freq == 0 else _contrast_eq(freq / larger_freq, freq, idf_score, tf_lift, idf_lift))
  return scores


def node_contrast_analysis(node: TaxoNode, parent: TaxoNode, siblings: List[TaxoNode] = [], tf_lift=1.5, idf_lift=1.5):
  """Compute term scores with contrastive analysis.

  Computation is based on popularity, discriminativeness, and
  idf.

  dis = min(tf / tf_parent, tf / tf_sibling)
  where tf_sibling is maximum tf in all siblings.

  term_score = dis * tf^tf_lift * idf^idf_lift

  Args:
    node: Current node.
    parent: Parent node.
    siblings: Optional list of sibling nodes.
    tf_lift: Multiplied with tf to give it more emphasize.
    idf_lift: Multiplied with idf to give it more emphasize.
  """
  pop = node.tf  # popularity
  inform = node.idf  # informativeness
  term_scores = dict()
  for term in node.terms:
    rel_p = parent.tf[term]  # assuming T \subset T_p
    rel_sib = -1
    for sib in siblings:
      if term in sib.terms and sib.tf[term] > rel_sib:
        rel_sib = sib.tf[term]
    if rel_sib > 0:  # term present in siblings
      distinct = (0 if pop[term] == 0 else np.min([(pop[term] / rel_p), (pop[term] / rel_sib)]))
    else:
      distinct = 0 if pop[term] == 0 else pop[term] / rel_p
    term_scores[term] = _contrast_eq(distinct, pop[term], inform[term], tf_lift, idf_lift)
  return term_scores
