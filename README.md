# Active Learning for Optimal Interventions

Code for paper: _Active learning for optimal intervention design in causal models (Nature Machine Intelligence, 2023)_

arXiv link: https://arxiv.org/abs/2209.04744

## Installation
Follow the two steps illustrated below

1. create a conda environment using `environment.yaml` (all dependencies are included; whole process takes about 5 min):
```
conda env create -f environment.yml
```
2. install the current package in editable mode inside the conda environment:
```
pip install -e .
```

## Examples on synthetic data
Run on a synthetic instance, e.g.:
```
python run.py --nnodes 5 --noise_level 1 --DAG_type path --std --a_size 2 --a_target 3 4 --acquisition greedy
```

Source code folder: `./optint/`

More examples given in: `./optint/notebook/test_multigraphs.ipynb`


## Examples on Perturb-CITE-Seq [1]

Source code folder: `./perturb-CITE-seq`

Notebooks for exploratory data analysis: `./perturb-CITE-seq/preprocess`

- download and extract data: `./perturb-CITE-seq/preprocess/screen_sanity_checks.ipynb`
- process data: `./perturb-CITE-seq/preprocess/process_data.ipynb`

Notebook for running the optimal intervention design task: `./perturb-CITE-seq/test.ipynb`

## Figures in the paper

Illustraive figures: made using mac keynotes

Pointers for nonillustrative figures:

- `./optint/notebook/test_ow.ipynb`: Fig. 3, Supplementary Fig. 2
- `./optint/notebook/test_convergence.ipynb`: Fig. 4
- `./optint/notebook/test_multigraphs.ipynb`: Fig. 5, Supplementary Fig. 4-7
- `./optint/notebook/test_moreacq.ipynb`: Supplementary Fig. 8
- `./optint/notebook/test_misspecgraphs.ipynb`: Supplementary Fig. 10
- `./perturb-CITE-seq/preprocess/screen_sanity_checks.ipynb`: Supplementary Fig. 11, 13, 14A
- `./perturb-CITE-seq/preprocess/process_data.ipynb`: Supplementary Fig. 12
- `./perturb-CITE-seq/preprocess/test_linearity.ipynb`: Supplementary Fig. 14C
- `./perturb-CITE-seq/test.ipynb`: Fig. 6, Supplementary Fig. 15-18
