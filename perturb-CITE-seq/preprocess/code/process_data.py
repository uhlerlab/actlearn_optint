import numpy as np
import pandas as pd
import pickle


ctrl_cells = pd.read_csv("../data/controlscreen_observational_samples_reduced.csv")
genes = ctrl_cells['GENE'].to_numpy()

with open("../save/pertub_dict.pkl", 'rb') as f:
	pertub_dict = pickle.load(f)

ptb_cells_id = np.concatenate([pertub_dict['Control'][a] if a != '' else [] for a in pertub_dict['Control'].keys()])
ptb_cells_full = pd.read_csv("../data/RNA_expression.csv", header=0, index_col=0, sep=',',
			usecols=np.concatenate([['GENE'], ptb_cells_id]))
ptb_cells = ptb_cells_full.loc[genes, :]
ptb_cells.to_csv("../data/controlscreen_interventional_samples_reduced.csv")
