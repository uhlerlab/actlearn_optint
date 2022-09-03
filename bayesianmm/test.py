import numpy as np

from bayesianmm.model import bayes_estimate_given_DAG
from bayesianmm.acquisition import *

from bayesianmm.eval import plot_acquisition


def test_passive(problem, opts):
	A = []
	Prob = []	
	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1})

	for _ in range(opts.T):
		a = np.random.uniform(-1,1,(problem.nnodes,1))

		idx = abs(a.flatten()).argsort()[:opts.nS]
		a[idx] = 0

		a = a / np.linalg.norm(a)
		A.append(a)
		batch, _, _ = problem.sample(a, opts.N)
		be_known_DAG.update_posterior(a, batch)
		Prob.append(be_known_DAG.prob_pad())

	return A, Prob


def test_active(problem, opts):
	A = []
	Prob = []
	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1})
	
	#assert W, "warm-up steps must be larger than 0!"
	for _ in range(opts.W):
		a = np.random.uniform(-1,1,(problem.nnodes,1))
		a = a / np.linalg.norm(a)
		A.append(a)
		batch, _, _ = problem.sample(a, opts.N)
		be_known_DAG.update_posterior(a, batch)
		Prob.append(be_known_DAG.prob_pad())

	acq_func = {
		'ivr': ivr_known_dag, 
		'ml2': ml2_known_dag, 
		'mestvr': mestvr_known_dag, 
		'svr': svr_known_dag,
		'l1vr': ivr_test, 
		'vr': vr_known_dag
	}.get(opts.acq, None)
	assert acq_func is not None, "Unsupported functions!"

	if opts.verbose:
		acq_vals = []

	for i in range(opts.T-opts.W):

		prob_pad = be_known_DAG.prob_pad()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])

		sigma_square = problem.sigma_square if opts.known_noise else np.zeros(problem.sigma_square.shape)
		if opts.acq == 'ivr':
			acquisition = acq_func(sigma_square, mean, var, problem.mu_target, opts.N, opts.comp)
		else:	
			acquisition = acq_func(sigma_square, mean, var, problem.mu_target, opts.N)
		
		try:
			# try two initial points
			a_jitter = a.reshape(-1)*np.random.uniform(0.8,1.2,(problem.nnodes,))
			x01 = np.maximum(np.minimum(a_jitter, 1), -1)
			
			x02 = problem.mu_target - np.matmul(mean, problem.mu_target)
			x02 = x02 / np.linalg.norm(x02)
			# x02 = np.maximum(np.minimum(a_jitter, 1), -1)

			# # let the initial int l2-norm = 1
			# x0 = a_jitter / np.linalg.norm(a_jitter)
			a1 = acquisition.optimize(x0=x01)# don't use gradient for now
			a2 = acquisition.optimize(x0=x02)
			if a1 is not None and a2 is not None:
				a1 = a1.reshape(-1,1)
				a2 = a2.reshape(-1,1)
				a = (a1, a2)[acquisition.evaluate(a1) > acquisition.evaluate(a2)]
			elif a1 is not None:
				a = a1.reshape(-1,1)
				print('2nd initialization (estimate) failed...')
			elif a2 is not None:
				a = a2.reshape(-1,1)
				print('1st initialization (last round) failed...')
			else:
				print('both initializations failed...')
		except UnboundLocalError:
			x0 = np.random.uniform(-1,1, problem.nnodes)
			# # let the initial int l2-norm = 1
			x0 = x0 / np.linalg.norm(x0)
			try:
				a = acquisition.optimize(x0=x0).reshape(-1,1) # don't use gradient for now
			except AttributeError:
				a = np.random.uniform(-1,1, problem.nnodes).reshape(-1,1)
				a = a / np.linalg.norm(a) 
		
		if opts.verbose:
			acq_vals.append(acquisition.evaluate(a))
			# print(f"acq at {i}={acquisition.evaluate(a)}")

		idx = abs(a.flatten()).argsort()[:opts.nS]
		a[idx] = 0

		A.append(a)

		batch, _, _ = problem.sample(a, opts.N)
		be_known_DAG.update_posterior(a, batch)
		Prob.append(be_known_DAG.prob_pad())

	if opts.verbose:
		plot_acquisition(opts, acq_vals)

	return A, Prob



def test_active_k(problem, opts):
	A = []
	Prob = []
	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1})
	
	#assert W, "warm-up steps must be larger than 0!"
	for _ in range(opts.W):
		a = np.random.uniform(-1,1,(problem.nnodes,1))
		a = a / np.linalg.norm(a)
		A.append(a)
		batch, _, _ = problem.sample(a, opts.N)
		be_known_DAG.update_posterior(a, batch)
		Prob.append(be_known_DAG.prob_pad())

	acq_func = {
		'ivr': ivr_known_dag, 
		'ml2': ml2_known_dag, 
		'mestvr': mestvr_known_dag, 
		'svr': svr_known_dag, 
		'vr': vr_known_dag
	}.get(opts.acq, None)
	assert acq_func is not None, "Unsupported functions!"

	if opts.verbose:
		acq_vals = []

	from copy import deepcopy
	mu_target = deepcopy(problem.mu_target)

	for i in range(opts.T-opts.W):
		if i % opts.k == 0 and i != 0:
			# a_star = np.dot(np.eye(problem.nnodes)-np.array(prob_pad['mean']), mu_target)
			a_star = np.dot(np.eye(problem.nnodes)-problem.B, mu_target)
			try:
				# k = np.where(a_star!=0)[0][0]
				# print(k)
				k += 1
				t = np.where(a_star!=0)[0][0:k]
				a_star_est[t] = a_star[t]
			except:
				a_star_est = np.zeros((problem.nnodes, 1))
				k = 1
				t = np.where(a_star!=0)[0][0:k]
				# print(k)
				a_star_est[t] = a_star[t]
			# mu_target = mu_target - np.dot(np.linalg.inv(np.eye(problem.nnodes)-np.array(prob_pad['mean'])), a_star_est)
			mu_target = mu_target - np.dot(problem.A, a_star_est)
			# # if np.linalg.norm(mu_target) > 0.01:
			# # 	mu_target /= np.linalg.norm(mu_target)
			# try:
			# 	k = np.where(mu_target!=0)[0][0]
			# 	# print(k)
			# 	if k != problem.nnodes - 1:
			# 		mu_target[k] = 0
			# 		# mu_target /= np.linalg.norm(mu_target)
			# except IndexError:
			# 	pass

		prob_pad = be_known_DAG.prob_pad()
		mean = np.array(prob_pad['mean'])
		var = np.array(prob_pad['var'])

		if opts.acq == 'ivr':
			acquisition = acq_func(problem.sigma_square, mean, var, mu_target, opts.N, opts.comp)
		else:	
			acquisition = acq_func(problem.sigma_square, mean, var, mu_target, opts.N)
		
		try:
			# try two initial points
			a_jitter = a.reshape(-1)*np.random.uniform(0.8,1.2,(problem.nnodes,))
			x01 = np.maximum(np.minimum(a_jitter, 1), -1)
			
			x02 = mu_target - np.matmul(mean, mu_target)

			# # let the initial int l2-norm = 1
			# x0 = a_jitter / np.linalg.norm(a_jitter)
			try:
				a1 = acquisition.optimize(x0=x01).reshape(-1,1) # don't use gradient for now
				a2 = acquisition.optimize(x0=x02).reshape(-1,1)
			except:
				print(acquisition.weight)
			a = (a1, a2)[acquisition.evaluate(a1) > acquisition.evaluate(a2)]

		except UnboundLocalError:
			x0 = np.random.uniform(-1,1, problem.nnodes)
			# # let the initial int l2-norm = 1
			# x0 = x0 / np.linalg.norm(x0)
			a = acquisition.optimize(x0=x0).reshape(-1,1) # don't use gradient for now
		
		if opts.verbose:
			acq_vals.append(acquisition.evaluate(a))
			# print(f"acq at {i}={acquisition.evaluate(a)}")

		idx = abs(a.flatten()).argsort()[:opts.nS]
		a[idx] = 0

		A.append(a)

		batch, _, _ = problem.sample(a, opts.N)
		be_known_DAG.update_posterior(a, batch)
		Prob.append(be_known_DAG.prob_pad())

	if opts.verbose:
		plot_acquisition(opts, acq_vals)

	return A, Prob




# def test_active_target_adjust(problem, opts):
# 	A = []
# 	Prob = []
# 	be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1})
	
# 	#assert W, "warm-up steps must be larger than 0!"
# 	for _ in range(opts.W):
# 		a = np.random.uniform(-1,1,(problem.nnodes,1))
# 		a = a / np.linalg.norm(a)
# 		A.append(a)
# 		batch, _, _ = problem.sample(a, opts.N)
# 		be_known_DAG.update_posterior(a, batch)
# 		Prob.append(be_known_DAG.prob_pad())

# 	for _ in range(opts.T-opts.W):
# 		prob_pad = be_known_DAG.prob_pad()
# 		mean = np.array(prob_pad['mean'])
# 		var = np.array(prob_pad['var'])
		
# 		acquisition = ivr_given_dag(problem.sigma_square, mean, var, problem.mu_target, opts.N, opts.alpha, opts.beta, opts.gamma, opts.zeta, opts.direct, opts.int)
# 		try:
# 			a_jitter = a.reshape(-1)*np.random.uniform(0.8,1.2,(problem.nnodes,))
# 			x0 = np.maximum(np.minimum(a_jitter, 1), -1)

# 			# # let the initial int l2-norm = 1
# 			# x0 = a_jitter / np.linalg.norm(a_jitter)
# 			a = acquisition.optimize(x0=x0).reshape(-1,1) # don't use gradient for now
# 		except UnboundLocalError:
# 			x0 = np.random.uniform(-1,1, problem.nnodes)
# 			# # let the initial int l2-norm = 1
# 			# x0 = x0 / np.linalg.norm(x0)
# 			a = acquisition.optimize(x0=x0).reshape(-1,1) # don't use gradient for now
		
# 		idx = abs(a.flatten()).argsort()[:opts.S]
# 		a[idx] = 0

# 		A.append(a)

# 		batch, _, _ = problem.sample(a, opts.N)
# 		be_known_DAG.update_posterior(a, batch)
# 		Prob.append(be_known_DAG.prob_pad())

# 	return A, Prob