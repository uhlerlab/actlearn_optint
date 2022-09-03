from sys import prefix
from graphical_models.classes.dags.dag import DAG
import numpy as np
from numpy.linalg import inv
import graphical_models as gm

from .dag import *

# generate DAG
def gen_dag(nnodes, DAG_type):

	DAG_gen = {
		'random': random_graph,
		'barabasialbert': barabasialbert_graph,
		'line': line_graph,
		'path': path_graph,
		'instar': instar_graph,
		'outstar': outstar_graph,
		'tree':  tree_graph,
		'complete': complete_graph,
		'chordal': chordal_graph,
		'rootedtree': rooted_tree_graph,
		'cliquetree': cliquetree_graph
	}.get(DAG_type, None)
	assert DAG_type is not None, 'Unsuppoted DAG type!'

	return DAG_gen(nnodes)	


# creating an object for the causal system and the target state
class mean_match(object):

	def __init__(self, nnodes, DAG_type, sigma_square, S, target=None, std=False, seed=1234, prefix_DAG=None):
		np.random.seed(seed)

		self.nnodes = nnodes
		self.sigma_square = sigma_square

		# read or generate DAG
		if prefix_DAG is None:
			self.DAG = gen_dag(nnodes, DAG_type)
		else:
			self.DAG = prefix_DAG
		if target is not None:
				self.DAG = ordered_DAG(self.DAG)

		# generate DAG weights
		self.weighted_DAG = gm.rand.rand_weights(self.DAG, weight_func)

		# create linear gaussian SCM
		self.eps_means = np.zeros(nnodes)	# wlog, set mean=0
		self.eps_cov = self.sigma_square*np.identity(nnodes)
		self.B = np.transpose(self.weighted_DAG.to_amat())
		self.A = inv(np.identity(nnodes)-self.B)
		self.mu = np.matmul(self.A, self.eps_means.reshape(-1,1)) 

		# standardize data to make each variable equal variance
		if std:
			scale = np.matmul(np.matmul(self.A, self.eps_cov), self.A.T).diagonal().reshape(-1,1)**(-0.5)
			self.eps_means = scale.reshape(-1)*self.eps_means
			self.eps_cov = scale * self.eps_cov * (scale.reshape(-1))
			self.sigma_square = self.eps_cov.diagonal()
			self.B = scale * self.B * ((1/scale).reshape(-1))
			self.A = scale * self.A * ((1/scale).reshape(-1))
			self.mu = scale * self.mu

		# generate target state
		if target is not None:
			idx = np.array(target)
		else:
			idx = np.random.choice(nnodes, S, replace=False)
		sgn = np.random.choice(2, S) * 2 -1
		a_idx = np.random.uniform(.25, 1, S) * sgn	# sparse matching intervention
		# make the matching int l2-norm=1
		a_idx = a_idx / np.linalg.norm(a_idx)
		self.a_target = np.zeros((nnodes, 1))
		self.a_target[idx] = a_idx.reshape(S, 1)

		# # not perturb sink nodes
		# for i in idx:
		# 	if len(self.DAG.children_of(i)) == 0:
		# 		self.a_target[i] = 0

		# calculate target mean
		self.mu_target = self.mu + np.matmul(self.A, self.a_target)

	# get N samples from intervention a
	def sample(self, a, N):
		eps = np.random.multivariate_normal(self.eps_means, self.eps_cov, N).reshape(self.nnodes, N)

		batch = np.dot(self.A, a + eps) 
		batch_loss = np.linalg.norm(batch-self.mu_target, axis=0)**2
		sample_loss = np.mean(batch_loss)

		return batch, batch_loss, sample_loss

	# evaluate intervention a from the underlying model
	def evaluate(self, a):
		mu = self.mu + np.dot(self.A, a) 
		loss = np.linalg.norm(mu-self.mu_target)**2 

		return loss.item()


# weight function
def weight_func(size):
	sgn = np.random.binomial(n=1, p=0.5, size=size) 
	
	rand = []
	for i in range(size):
		if sgn[i]==0:
			# recommended low & high: e^{-1/p} & e^{1/p}
			rand.append(np.random.uniform(low=-1, high=-0.25))
		else:
			rand.append(np.random.uniform(low=0.25, high=1))
		
	return rand

