# Active Learning for Optimal Interventions

Code for paper: _Active learning for optimal intervention design in causal models_

arXiv link: coming soon...

## Installation
Download code and run in command line:
```
pip install -e .
```

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

- `./optint/notebook/test_ow.ipynb`: Fig. 3, S2
- `./optint/notebook/test_convergence.ipynb`: Fig. 4
- `./optint/notebook/test_multigraphs.ipynb`: Fig. 5, S4-7
- `./perturb-CITE-seq/preprocess/screen_sanity_checks.ipynb`: Fig. S8, S10, S11A
- `./perturb-CITE-seq/preprocess/process_data.ipynb`: Fig. S9
- `./perturb-CITE-seq/preprocess/test_linearity.ipynb`: Fig. S11C
- `./perturb-CITE-seq/test.ipynb`: Fig. 6, S12-15
