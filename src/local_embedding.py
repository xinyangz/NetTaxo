import logging
from pathlib import Path

from tqdm.auto import tqdm

import config
from sgns import SGNS_C
from taxonomy import TaxoNode, parse_prefix

logger = logging.getLogger(__name__)


def _prepare_local_corpus(data_dir, documents, weights, level, order):
  if weights is None:
    weights = {paper_id: 1 for paper_id in documents.keys()}

  # rescale weights to 1...10
  w_min = min(weights.values())
  w_max = max(weights.values())
  W = dict()
  for paper_id, w in weights.items():
    if w_max > w_min:
      W[paper_id] = int(1 + ((w - w_min) / (w_max - w_min) * 9))
    else:
      W[paper_id] = 1

  if level == 1:
    text_file_p = Path(data_dir, "text_documents.txt")
    ind_file_p = Path(data_dir, "indices.txt")
    text_file_c = Path(data_dir, "intermediate/local_text_lv0.txt")
    corpus_file_c = Path(data_dir, "intermediate/local_corpus_lv0.txt")
    ind_file_c = Path(data_dir, "intermediate/local_ind_lv0.txt")
  else:
    text_file_p = Path(data_dir, f"intermediate/local_corpus_lv0.txt")
    ind_file_p = Path(data_dir, "intermediate/local_ind_lv0.txt")
    text_file_c = Path(data_dir, f"intermediate/local_text_{level}_{order}_{config.unique_id}.txt")
    corpus_file_c = Path(data_dir, f"intermediate/local_corpus_{level}_{order}_{config.unique_id}.txt")
    ind_file_c = Path(data_dir, f"intermediate/local_ind_{level}_{order}_{config.unique_id}.txt")
  if corpus_file_c.is_file():
    return corpus_file_c
  with open(text_file_p,"r") as f, \
       open(ind_file_p, "r") as f_ind, \
       open(text_file_c, "w") as f_out, \
       open(corpus_file_c, "w") as f_corpus_out, \
       open(ind_file_c, "w") as f_ind_out:
    for ind in tqdm(f_ind):
      ind = ind.strip()
      line = next(f)
      if ind in documents:
        f_out.write(line)
        f_ind_out.write(ind + "\n")
        f_corpus_out.write(line * W[ind])
  return corpus_file_c


def local_embedding(node: TaxoNode, data_dir):
  prefix_name = node.prefix.replace("/", "_")
  prefix = parse_prefix(node.prefix)
  level = len(prefix)
  order = prefix[-1]

  if level == 1:
    unique_id = "lv0"
  else:
    unique_id = config.unique_id

  corpus_file = _prepare_local_corpus(data_dir, node.docs, node.doc_prior, level, order)

  if level == 1:
    word_syn0_file = Path(data_dir, f"intermediate/word_emb_{prefix_name}_{unique_id}.syn0.txt")
    net_syn0_file = Path(data_dir, f"intermediate/net_emb_{prefix_name}_{unique_id}.syn0.txt")
    if word_syn0_file.is_file() and net_syn0_file.is_file():
      logger.info("Load level 0 embedding w/o training")
      word_emb = SGNS_C(dim=100)
      word_emb.load(word_syn0_file)
      net_emb = SGNS_C(dim=100)
      net_emb.load(net_syn0_file)
      return word_emb, net_emb

  syn0_load = Path(data_dir, f"intermediate/word_emb_0_lv0.syn0.txt")
  syn1_load = Path(data_dir, f"intermediate/word_emb_0_lv0.syn1.txt")

  word_emb = SGNS_C(
      dim=100,
      n_thread=16,
      window=5,
      negative=5,
      n_iter=5,
      lam_base=0,
      lam_net=0,
      use_net=False,
  )

  output_name = Path(data_dir, f"intermediate/word_emb_{prefix_name}_{unique_id}")
  if level == 1:
    word_emb.train(corpus_file, output=output_name)
  else:
    word_emb.n_iter = 2
    word_emb.train(corpus_file, output=output_name, syn0=syn0_load, syn1=syn1_load, alpha=0.001)

  syn0_load = Path(data_dir, f"intermediate/net_emb_0_lv0.syn0.txt")
  syn1_load = Path(data_dir, f"intermediate/net_emb_0_lv0.syn1.txt")

  net_corpus_file = Path(data_dir, f"intermediate/net_context_{unique_id}.txt")
  net_emb = SGNS_C(
      dim=100,
      n_thread=16,
      window=5,
      negative=5,
      n_iter=5,
      lam_base=0,
      lam_net=1,
      use_net=True,
  )
  output_name = Path(data_dir, f"intermediate/net_emb_{prefix_name}_{unique_id}")
  if level == 1:
    net_emb.train(corpus_file, net_corpus_file=net_corpus_file, output=output_name)
  else:
    net_emb.n_iter = 2
    net_emb.train(
        corpus_file,
        net_corpus_file=net_corpus_file,
        output=output_name,
        alpha=0.001,
        syn0=syn0_load,
        syn1=syn1_load,
    )
  return word_emb, net_emb


def joint_local_embedding(node, data_dir):
  prefix_name = node.prefix.replace("/", "_")
  prefix = parse_prefix(node.prefix)
  level = len(prefix)
  order = prefix[-1]
  if level == 1:
    unique_id = "lv0"
  else:
    unique_id = config.unique_id
  corpus_file = _prepare_local_corpus(data_dir, node.docs, node.doc_prior, level, order)
  net_corpus_file = Path(data_dir, f"intermediate/net_context_all_{unique_id}.txt")
  joint_emb = SGNS_C(
      dim=100,
      n_thread=16,
      window=5,
      negative=5,
      n_iter=5,
      lam_base=0,
      lam_net=0.5,
      use_net=True,
  )
  syn0_load = Path(data_dir, f"intermediate/word_emb_0_lv0.syn0.txt")
  syn1_load = Path(data_dir, f"intermediate/word_emb_0_lv0.syn1.txt")
  output_name = Path(data_dir, f"intermediate/joint_emb_{prefix_name}_{config.unique_id}")
  if level == 1:
    joint_emb.train(corpus_file, net_corpus_file=net_corpus_file, output=output_name)
  else:
    joint_emb.n_iter = 2
    joint_emb.train(corpus_file,
                    net_corpus_file=net_corpus_file,
                    output=output_name,
                    alpha=0.005,
                    syn0=syn0_load,
                    syn1=syn1_load)
  return joint_emb
