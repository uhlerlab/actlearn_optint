import numpy as np
from model import bayes_estimate_given_DAG
from acquisition import ivr_known_dag, ml2_known_dag, svr_known_dag, mestvr_known_dag


# i'm calling this random since it differs from passive in the sense that it does make use of observational data
def test_random(problem, opts):
	A = []
	Prob = []	
	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1}, preprocess=opts.preprocess)

	# prior: observational data
	a = np.zeros((problem.nnodes,1))
	batch = problem.sample(a, opts.N0)
	be_known_DAG.update_posterior(a, batch)
	A.append(a)
	Prob.append(be_known_DAG.prob_pad())

	# posterior: interventional data
	for _ in range(opts.T-1):
		problem.update_pool(min_sample_size=opts.N)
		idx = np.random.choice(range(len(problem.a_pool)))
		a = problem.a_pool[idx]
		batch = problem.sample(a, opts.N)

		be_known_DAG.update_posterior(a, batch)
		A.append(a)
		Prob.append(be_known_DAG.prob_pad())

		if opts.unique:
			problem.reduce_pool(a)

	if opts.unique:
		problem.restore_pool()

	return A, Prob


def test_active(problem, opts):
	A = []
	Prob = []
	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1}, preprocess=opts.preprocess)
	
	# prior: observational data
	a = np.zeros((problem.nnodes,1))
	batch = problem.sample(a, opts.N)
	be_known_DAG.update_posterior(a, batch)
	A.append(a)
	Prob.append(be_known_DAG.prob_pad())

	# posterior: interventional data
	for _ in range(opts.T-1):
		sigma_square = be_known_DAG.prob['sigma_square']
		sigma_square = np.zeros(sigma_square.shape)
		prob_pad = be_known_DAG.prob_pad()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])
		if opts.acq == 'ivr':
			acquisition = ivr_known_dag(sigma_square, mean, var, problem.mu_target, opts.N, opts.method)
		elif opts.acq == 'ml2':
			acquisition = ml2_known_dag(sigma_square, mean, var, problem.mu_target, opts.N)
		elif opts.acq == 'svr':
			acquisition = svr_known_dag(sigma_square, mean, var, problem.mu_target, opts.N)
		elif opts.acq == 'mestvr':
			acquisition = mestvr_known_dag(sigma_square, mean, var, problem.mu_target, opts.N)
		# elif opts.acq == 'svr':
		# 	acquisition = svr_known_dag(sigma_square, mean, var, problem.mu_target, opts.N, np.hstack(problem.a_pool))
		
		problem.update_pool(opts.N)
		a = acquisition.optimize(problem.a_pool)
		batch = problem.sample(a, opts.N)

		be_known_DAG.update_posterior(a, batch)
		A.append(a)
		Prob.append(be_known_DAG.prob_pad())

		if opts.unique:
			problem.reduce_pool(a)

	if opts.unique:
		problem.restore_pool()

	return A, Prob