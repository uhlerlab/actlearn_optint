from bdb import GENERATOR_AND_COROUTINE_FLAGS
import os
import json
import pickle
import argparse
#import pprint as pp
from argparse import Namespace
from xml.etree.ElementInclude import default_loader
import numpy as np
from tqdm import tqdm
import timeit

from optint.data import synthetic_instance, gen_dag
from optint.test import test_passive, test_active


def run(problem, opts):

	if opts.time:
		start = timeit.default_timer()

	test = {
		True: test_active,
		False: test_passive 
	}.get(opts.active)

	Actions = []
	Probs = []

	for r in tqdm(range(opts.R)):
		action, prob = test(problem, opts)

		Actions.append(action)
		Probs.append(prob)

	if opts.time:
		stop = timeit.default_timer()
		return Actions, Probs, stop - start

	else:
		return Actions, Probs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# synthetic instances
	parser.add_argument('--num_instances', type=int, default=10, help='Number of instances generated (int).')
	parser.add_argument('--diff_DAGs', action='store_true', help='Use the same DAG across instances (store true)')
	parser.add_argument('--nnodes', type=int, help='<Required> Number of nodes in the graph (int).', required=True)
	parser.add_argument('--noise_level', type=float, help='<Required> Noise level of observations (float).', required=True)
	parser.add_argument('--DAG_type', type=str, help='<Required> DAG type (str).', required=True)
	parser.add_argument('--std', action='store_true', help='Standardize to make equal variance (store_true).')
	parser.add_argument('--a_size', type=int, help='<Required> Number of intervention targets (int).', required=True)
	parser.add_argument('--a_target', type=int, nargs='+', help='Intervention targets (ints).')

	# learning method
	parser.add_argument('--acquisition', type=str, nargs='+', help='Type of acquisition functions used for active learning (strs).')
	parser.add_argument('--civ_measure', type=str, default='unif', help='Type of measure used by CIV')
	parser.add_argument('--sample_size', type=int, default=1, help='Number of samples acquired from each intervention')
	parser.add_argument('--total_time_step', type=int, default=50, help='Total number of time steps')
	parser.add_argument('--warm_up_step', type=int, default=0, help='Number of warm up steps')
	parser.add_argument('--repeat_runs', type=int, default=20, help='Number of repeated runs')
	parser.add_argument('--unknown_variance', action='store_false', help='Condition on known variances (store_false)')

	# saved directory
	parser.add_argument('--save_dir', default='results/', help="Saving directory.")

	# read args
	args = parser.parse_args()
	if args.a_target is not None:
		assert len(args.a_target)==args.a_size, 'Unmatched numbers of perturbation targets.'
	assert args.total_time_step > args.warm_up_step, 'Total time steps must be larger than warm up steps.'

	# saved file name
	std = '-std' if args.std else ''
	savedir = os.path.join(
		args.save_dir, 
		'{}{}_nnodes{}_noise{:.2f}_S{}_target{}'.format(args.DAG_type, std, args.nnodes, args.noise_level, args.a_size, args.a_target)
		)
	if not os.path.isdir(savedir):
		os.makedirs(savedir)

	# options for testing
	opts = Namespace(
		n=args.sample_size, 
		T=args.total_time_step, 
		W=args.warm_up_step, 
		R=args.repeat_runs, 
		known_noise=args.unknown_variance, 
		measure=args.civ_measure
		)
	with open('{}/args.json'.format(savedir), 'w') as f:
		json.dump(vars(opts), f, indent=True)

	# run algorithms
	problems = []
	As = []
	Probs = []
	graph = None if args.diff_DAGs else gen_dag(nnodes=args.nnodes, DAG_type=args.DAG_type)
	for i in range(args.num_instances):
		print('Graph {}'.format(i+1))

		problem = synthetic_instance(
			nnodes=args.nnodes,
			DAG_type=args.DAG_type,
			sigma_square=args.noise_level * np.ones(args.nnodes),
			a_size=args.a_size,
			a_target_nodes=args.a_target,
			std=args.std,
			prefix_DAG=graph
		)
		problems.append(problem)

		A = {}
		Prob = {}
		
		opts.active = False
		A['passive'], Prob['passive'] = run(problem, opts)

		opts.active = True
		for acq in args.acquisition:
			opts.acq = acq
			A[acq], Prob[acq] = run(problem, opts)

		As.append(A)
		Probs.append(Prob)

	# save results
	with open('{}/problems.pkl'.format(savedir), 'wb') as f:
		pickle.dump(problems, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open('{}/As.pkl'.format(savedir), 'wb') as f:
		pickle.dump(As, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open('{}/Probs.pkl'.format(savedir), 'wb') as f:
		pickle.dump(Probs, f, protocol=pickle.HIGHEST_PROTOCOL)		

