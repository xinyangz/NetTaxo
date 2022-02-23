"""Command line arguments and config"""

import argparse
from datetime import datetime
from pathlib import Path

unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
N_TOP_TERMS = 500
TOP_TERMS_PCT = 0.2
TOP_MOTIF_PCT = 0.2
LEVEL_DECAY = 0.9
N_ANCHOR_TERMS = 50


def handle_args(s=""):
  parser = argparse.ArgumentParser(description="Automatic Taxonomy Generation")
  parser.add_argument("--base_name", default="exp", help="A short comment line to distinguish experiments.")
  parser.add_argument("-d", "--data_dir", default="data/dblp-5area/", type=Path, help="path to dataset")
  parser.add_argument("--output_dir", default="output/", type=Path, help="path to output")
  parser.add_argument("--log_dir", default="log/", type=Path, help="path to log")
  parser.add_argument(
      "--levels",
      nargs="+",
      default=[5, 4],
      type=int,
      help="Number of nodes per taxonomy layer."\
           "Seperate using space. E.g. --levels 5 4 3",
  )
  parser.add_argument("--tf_lift", default=1.5, type=float, help="multiplier to tf in discriminative analysis")
  parser.add_argument("--idf_lift", default=1.5, type=float, help="multiplier to idf in discriminative analysis")
  parser.add_argument("--damping", default=0.8, type=float, help="damping factor for personalized pagerank")
  if s:
    args = parser.parse_args(s)
  else:
    args = parser.parse_args()
  return args
