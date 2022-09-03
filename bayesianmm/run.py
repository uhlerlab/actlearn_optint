import os
import json
import pickle
import argparse
#import pprint as pp
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bayesianmm.data import mean_match
from bayesianmm.test import test_passive, test_active, test_active_k

from bayesianmm.visualize import draw, draw_spectrum
from bayesianmm.eval import *


def run(problem, opts):
	test = {
		True: test_active,
		False: test_passive,
		'de': test_active_k
	}.get(opts.active)

	Actions = []
	Probs = []

	for r in tqdm(range(opts.R)):
		action, prob = test(problem, opts)

		Actions.append(action)
		Probs.append(prob)

	return Actions, Probs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# problem-based args
	parser.add_argument('--nnodes', type=int, help="<Required> Number of nodes in the graph (int).", required=True)
	parser.add_argument('--noise_level', type=float, help="<Required> Noise level of observations (float).", required=True)
	parser.add_argument('--DAG_type', type=str, help="<Required> DAG type (str).", required=True)
	parser.add_argument('--num_DAG', type=int, default=10, help="Number of DAGs generated (int).")
	parser.add_argument('--std', action='store_true', help='Standardize to make equal variance (store_true).')
	parser.add_argument('--S', type=int, help='<Required> Number of perturbation targets (int).', required=True)
	parser.add_argument('--target', type=int, nargs='+', help='Perturbation targets (ints).')

	# policy-based args
	parser.add_argument('--acquisition', type=str, nargs='+', help="Type of acquisition functions used for active learning (strs).")

	# saved results
	parser.add_argument('--save_dir', default='results/ivr_given_DAG/target_diff', help="Saving directory.")
	parser.add_argument('--savemodel', action='store_true', help='Save the models (store_true).')
	parser.add_argument('--saveints', action='store_true', help='Save the picked interventions (store_true).')

	args = parser.parse_args()
	if args.target is not None:
		assert len(args.target)==args.S, 'Unmatched numbers of perturbation targets.'

	std = '-std' if args.std else ''
	savedir = os.path.join(args.save_dir, '{}{}_nnodes{}_noise{:.2f}_S{}_target{}'.format(args.DAG_type, std, args.nnodes, args.noise_level, args.S, args.target))
	if not os.path.isdir(savedir):
		os.makedirs(savedir)

	# options for testing
	opts = Namespace(N=1, T=50, W=0, R=50, nS=0, verbose=False)
	with open(f'{savedir}/args.json', 'w') as f:
		json.dump(vars(opts), f, indent=True)

	MSEs = []
	for i in range(args.num_DAG):
		print(f'Graph {i+1}')

		problem = mean_match(
			nnodes=args.nnodes,
			sigma_square=args.noise_level * np.ones(args.nnodes),
			DAG_type=args.DAG_type,
			S=args.S,
			target=args.target,
			std=args.std
		)

		draw(
			problem.DAG,
			colored_set = set(i for i in range(problem.nnodes) if problem.a_target[i]!=0), 
			affected_set=set().union(*[problem.DAG.descendants_of(i) for i in range(problem.nnodes) if problem.a_target[i]!=0]),
			edge_weights=problem.B.T,
			savefile=f'{savedir}/run{i+1}-DAG.png'
		)

		draw_spectrum(problem.A, problem.B, savefile=f'{savedir}/run{i+1}-spectrum.png')

		A = {}
		Prob = {}
		
		opts.active = False
		A['passive'], Prob['passive'] = run(problem, opts)

		opts.active = True
		for acq in args.acquisition:
			opts.acq = acq
			A[acq], Prob[acq] = run(problem, opts)

		mses = plot_matching_intervention(
			problem,
			opts,
			f'{savedir}/run{i+1}-matchingints.png',
			**Prob
		)
		MSEs.append(mses)

		if args.saveints:
			plot_picked_interventions(problem, opts, f'{savedir}/run{i+1}-pickedints.png', **A);

		if args.savemodel:
			with open(f'{savedir}/run{i+1}.pkl', 'wb') as f:
				pickle.dump([problem, A, Prob], f, protocol=pickle.HIGHEST_PROTOCOL)

	draw(
		problem.DAG, 
		colored_set = set(i for i in range(problem.nnodes) if problem.a_target[i]!=0),
		affected_set=set().union(*[problem.DAG.descendants_of(i) for i in range(problem.nnodes) if problem.a_target[i]!=0]),
		savefile=f'{savedir}/DAG.png'
	)

	plt.clf()
	plt.figure(figsize=(5,3))
	plt.yscale('log')
	for k in ['passive']+args.acquisition:
		mean = np.array([np.array(MSEs[i][k]).mean(axis=0) for i in range(args.num_DAG)]).mean(axis=0)
		std =np.array([np.array(MSEs[i][k]).std(axis=0) for i in range(args.num_DAG)]).mean(axis=0)
		plt.plot(range(opts.T), mean, label=k)
		plt.fill_between(range(opts.T), mean - 0.1*std, mean + 0.1*std, alpha=.2)
	plt.legend()
	plt.xlabel('Round')
	plt.ylabel('Mean Absolute Error')
	plt.title(f'Matching Intervention (Average {args.num_DAG} Graphs)')
	plt.tight_layout()
	plt.savefig(f'{savedir}/matchingints.png')
	plt.close()

	plt.clf()
	plt.figure(figsize=(len(args.acquisition)+1,3))
	data = []
	for k in ['passive']+args.acquisition:
		data.append([np.array(MSEs[i][k]).mean(axis=0)[-1] for i in range(args.num_DAG)])
	plt.boxplot(data, labels=['passive']+args.acquisition);
	plt.ylabel('Mean Absolute Error')
	plt.title(f'Round {opts.T} ({args.num_DAG} Graphs)')
	plt.tight_layout()
	plt.savefig(f'{savedir}/pickedints.png')
	plt.close()
