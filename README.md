# NetTaxo
NetTaxo: Automated Topic Taxonomy Construction from Text-Rich Network

## Run the Experiment

### Requirements
```bash
python>=3.7
spherecluster
scikit-learn<=0.22
joblib
numba
pydot
python-igraph
scipy
tqdm
```

### Run
```bash
make
python src/build_taxonomy.py --data_dir data/dblp-5area
```

Output will be saved to `--output_dir`. A taxonomy visualization, a taxonomy dump gz file, and the taxonomy nodes will be saved. Each folder represents a taxonomy node, with the term score distribution and document score distribution saved into two files.

## Data
[Download](https://www.dropbox.com/s/5roebz5yy8tim5x/data.zip?dl=0) and unzip the data into `/data`.

Please refer to `data/dblp-5area` for data formats.

For use on custom datasets, format the data according to the example dataset.
Motif matching requires additional coding, as motif patterns might be different from dataset to dataset.
Refer to `src/motif_embed.py` for motif matcher examples.
Write custom motif matchers, then include them in the main file `src/build_taxonomy.py`.

