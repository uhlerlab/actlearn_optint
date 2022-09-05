import numpy as np
from model import linearSCM
from acquisition import *


def test_random(problem, opts):
	A = []
	Prob = []	
	model = linearSCM(problem.DAG, {'pot_vec': 0, 'info_mat': 1}, preprocess=opts.preprocess)

	# prior: observational data
	a = np.zeros((problem.nnodes,1))
	batch = problem.sample(a, opts.N0)
	model.update_posterior(a, batch)
	A.append(a)
	Prob.append(model.prob_padded())

	# posterior: interventional data
	for _ in range(opts.T-1):
		idx = np.random.choice(range(len(problem.a_pool)))
		a = problem.a_pool[idx]
		batch = problem.sample(a, opts.N)

		model.update_posterior(a, batch)
		A.append(a)
		Prob.append(model.prob_padded())

		if opts.unique:
			problem.reduce_pool(a)

	if opts.unique:
		problem.restore_pool()

	return A, Prob


def test_active(problem, opts):
	A = []
	Prob = []
	model = linearSCM(problem.DAG, {'pot_vec': 0, 'info_mat': 1}, preprocess=opts.preprocess)
	
	# prior: observational data
	a = np.zeros((problem.nnodes,1))
	batch = problem.sample(a, opts.N)
	model.update_posterior(a, batch)
	A.append(a)
	Prob.append(model.prob_padded())

	# posterior: interventional data
	for _ in range(opts.T-1):
		sigma_square = model.prob['sigma_square']
		sigma_square = np.zeros(sigma_square.shape)
		prob_pad = model.prob_padded()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])
		if opts.acq == 'civ':
			acquisition = civ_acq(sigma_square, mean, var, problem.mu_target, opts.N, opts.measure)
		elif opts.acq == 'greedy':
			acquisition = greedy_acq(sigma_square, mean, var, problem.mu_target, opts.N)
		elif opts.acq == 'cv':
			acquisition = cv_acq(sigma_square, mean, var, problem.mu_target, opts.N)
		elif opts.acq == 'maxv':
			acquisition = maxv_acq(sigma_square, mean, var, problem.mu_target, opts.N)

		a = acquisition.optimize(problem.a_pool)
		batch = problem.sample(a, opts.N)

		model.update_posterior(a, batch)
		A.append(a)
		Prob.append(model.prob_padded())

		if opts.unique:
			problem.reduce_pool(a)

	if opts.unique:
		problem.restore_pool()

	return A, Prob