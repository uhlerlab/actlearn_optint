import numpy as np
from tqdm import tqdm
import pickle

from data import mean_match
from model import bayes_estimate_given_DAG
from acquisition import ivr_given_dag

p = 3	# number of nodes
sigma_square = 0.1	# noise level
S = 3	# sparse true intervention
N = 10	# number of samples
W = 0
num_pts = 10
int_pts = 100
sample_pts = 500

problem = mean_match(nnodes=p, sigma_square=sigma_square, S=S, DAG_type='barabasialbert')

be_known_DAG = bayes_estimate_given_DAG(problem.DAG, {'pot_vec': 0, 'info_mat': 1})
for _ in range(W):
	a = np.random.uniform(-1,1,(p,1))
	batch, _, _ = problem.sample(a, N)
	be_known_DAG.update_posterior(a, batch)

prob_pad = be_known_DAG.prob_pad()
mean = np.array(prob_pad['mean'])
var = np.array(prob_pad['var'])
sigma_square_vec = sigma_square*np.ones((p,1))
acquisition = ivr_given_dag(sigma_square_vec, mean, var, problem.mu_target, N)

A = [np.random.uniform(-1,1,(p,1)) for _ in range(num_pts)]
ivr_A = [acquisition.evaluate(a) for a in A]

int_A = [np.random.uniform(-1,1,(p,1)) for _ in range(int_pts)]
var_A = []
for i in range(10):
	a = A[i]

	from copy import deepcopy
	be = deepcopy(be_known_DAG)

	B = be.prob_pad()['mean']
	bar_x = np.matmul(np.linalg.inv(np.eye(p)-B), a)
	batch = np.concatenate([bar_x for _ in range(N)])

	be.update_posterior(a, batch)

	int_var = 0

	from tqdm import tqdm
	for j in tqdm(range(int_pts)):

		samples = []

		for m in range(sample_pts):
			eps = np.random.multivariate_normal(np.zeros(p), sigma_square*np.eye(p)).reshape(p,1)
			
			B = np.zeros((p,p))
			for k in range(p):
				B[k] = np.random.multivariate_normal(be.prob_pad()['mean'][k], be.prob_pad()['var'][k])
			
			samples.append(((int_A[j]+eps-np.matmul(np.eye(p)-B, problem.mu_target))**2).sum())
		
		int_var += np.array(samples).var()

	var_A.append(int_var/int_pts)


with open("results/debug_ivr/p{}_W{}_{}_{}.pkl".format(p, W, int_pts, sample_pts), "wb") as file:
	pickle.dump({'ivr_A': ivr_A, 'var_A': var_A}, file, protocol=pickle.HIGHEST_PROTOCOL)