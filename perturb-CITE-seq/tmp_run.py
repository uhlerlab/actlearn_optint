from generate import mean_match
from test import test_random, test_active
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
import eval
import os
from tqdm import tqdm

def save_fig(path, file):
	if not os.path.isdir(path):
		os.makedirs(path)
	plt.savefig(os.path.join(path, file))

	
def temp(target):
	seed = 1234
	np.random.seed(seed)

	problem = mean_match(combination=False, seed=seed, target=target)
	
	opts = Namespace(preprocess = 'nonzero', N0=100, N=10, T=10, unique=False)
	R = 50

	A = []
	Prob = []
	for r in range(R):
		a, prob = test_random(problem, opts)
		A.append(a)
		Prob.append(prob)

	opts.acq = 'ivr'
	opts.method = 'sphere'
	A_0 = []
	Prob_0 = []
	for r in range(R):
		a, prob = test_active(problem, opts)
		A_0.append(a)
		Prob_0.append(prob)

	opts.acq = 'ml2'
	A_1 = []
	Prob_1 = []
	for r in range(R):
		a, prob = test_active(problem, opts)
		A_1.append(a)
		Prob_1.append(prob)

	opts.acq = 'svr'
	A_3 = []
	Prob_3 = []
	for r in range(R):
		a, prob = test_active(problem, opts)
		A_3.append(a)
		Prob_3.append(prob)

	plt.clf()
	eval.plot_guess(problem, opts, R, best=False, ivr=Prob_0, ml2=Prob_1, random=Prob, svr=Prob_3)
	save_fig('fig/'+target.split('_')[1], 'guess.png')

	plt.clf()
	eval.plot_guess(problem, opts, R, best=True, CIV=Prob_0, Greedy=Prob_1, Random=Prob, svr=Prob_3)
	save_fig('fig/'+target.split('_')[1], 'guess-best.png')

	plt.clf()
	eval.plot_mean(problem, opts, R, ivr=Prob_0, ml2=Prob_1, random=Prob, svr=Prob_3)
	save_fig('fig/'+target.split('_')[1], 'mean.png')

	plt.clf()
	eval.plot_mean(problem, opts, R, best=True, CIV=Prob_0, Greedy=Prob_1, Random=Prob, svr=Prob_3)
	save_fig('fig/'+target.split('_')[1], 'mean-best.png')

	plt.clf()
	eval.plot_a_hit(problem, opts, R, CIV=A_0, Greedy=A_1, Random=A, Svr=A_3)
	save_fig('fig/'+target.split('_')[1], 'hit.png')

	plt.clf()
	eval.plot_hit(problem, opts, R, CIV=Prob_0, Greedy=Prob_1, Random=Prob, Svr=Prob_3)
	save_fig('fig/'+target.split('_')[1], 'projguess-hit.png')


if __name__ == '__main__':
	for target in tqdm(['ptb_SP100', 'ptb_HLA-A', 'ptb_TGFB1', 'ptb_B2M', 'ptb_STAT3', 'ptb_CTSD', 'ptb_SOX4', 'ptb_NPC1', 'ptb_NPC2', 'ptb_CDK4', 'ptb_CTSB', 'ptb_CCR10', 'ptb_CDK6', 'ptb_HLA-B', 'ptb_EIF3K', 'ptb_SEC11C', 'ptb_SAT1', 'ptb_CDH19', 'ptb_CD59', 'ptb_STAT1', 'ptb_IRF3', 'ptb_IFNGR1', 'ptb_SERPINE2', 'ptb_CTSA', 'ptb_CD58', 'ptb_MYC', 'ptb_RB1', 'ptb_IFNGR2', 'ptb_SMAD4', 'ptb_HLA-C', 'ptb_FOS', 'ptb_DNMT1', 'ptb_HLA-E', 'ptb_NGFR', 'ptb_LAMP2', 'ptb_CXCR4']):
		temp(target)