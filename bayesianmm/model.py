import numpy as np
from numpy.linalg import inv

# bayesian model for the weight matrix B
#	bayes_estimate_given_DAG: update parameter posterior on an input DAG
#	bayes_estimate_one_DAG: update parameter posterior on one updated DAG

class bayes_estimate_given_DAG():

	def __init__(self, DAG, hparams):
		self.DAG = DAG
		self.p = DAG.nnodes
		self.hparams = hparams

		self.prob = self._prior()

	def _prior(self):
		pot_vec = [self.hparams['pot_vec'] * np.ones(self.DAG.indegree_of(i)) for i in range(self.p)]
		info_mat = [self.hparams['info_mat'] * np.eye(self.DAG.indegree_of(i)) for i in range(self.p)]

		self.prob = {'pot_vec': pot_vec, 'info_mat': info_mat}
		return self.prob


	def update_posterior(self, a, batch):
		
		for i in range(self.p):
			parents_idx = list(self.DAG.parents_of(i))
			parents_idx.sort()

			X_par = batch[parents_idx, :]
			X_i_correct = batch[i, :] - a[i]

			self.prob['info_mat'][i] += np.matmul(X_par, X_par.T)
			self.prob['pot_vec'][i] += np.matmul(X_i_correct, X_par.T)

	def prob_pad(self):
		mean = [np.zeros(self.p) for _ in range(self.p)]
		var = [np.zeros((self.p, self.p)) for _ in range(self.p)]

		for i in range(self.p):
			parents_idx = list(self.DAG.parents_of(i))
			parents_idx.sort()

			var[i][np.ix_(parents_idx,parents_idx)] = inv(self.prob['info_mat'][i])
			mean[i][parents_idx] = np.matmul(var[i][np.ix_(parents_idx,parents_idx)], self.prob['pot_vec'][i])

		prob_pad = {'mean': mean, 'var': var}

		return prob_pad


class bayes_estimate_one_DAG():

	def __init__(self, hparams):
		self.hparams = hparams

		self.A = []
		self.Batch = []

	def _update_DAG(self):
		pass

	def _prior(self, DAG):
		p = DAG.nnodes
		pot_vec = [self.hparams['pot_vec'] * np.ones(DAG.indegree_of(i)) for i in range(p)]
		info_mat = [self.hparams['info_mat'] * np.eye(DAG.indegree_of(i)) for i in range(p)]
		
		return {'pot_vec': pot_vec, 'info_mat': info_mat}

	def update_posterior(self, a, batch):
		self.A.append(a)
		self.Batch.append(batch)

		self.DAG = self._update_DAG()
		self.prob = self._prior(self.DAG)

		for i in range(self.DAG.nnodes):
			parents_idx = list(self.DAG.parents_of(i))
			parents_idx.sort()

			for j, a in enumerate(self.A):
				X_par = self.Batch[j][parents_idx, :]
				X_i_correct = self.Batch[j][i, :] - a[i]

				self.prob['info_mat'][i] += np.matmul(X_par, X_par.T)
				self.prob['pot_vec'][i] += np.matmul(X_i_correct, X_par.T)

	def prob_pad(self):
		p = self.DAG.nnodes
		mean = [np.zeros(p) for _ in range(p)]
		var = [np.zeros((p, p)) for _ in range(p)]

		for i in range(p):
			parents_idx = list(self.DAG.parents_of(i))
			parents_idx.sort()

			var[i][np.ix_(parents_idx,parents_idx)] = inv(self.prob['info_mat'][i])
			mean[i][parents_idx] = np.matmul(var[i][np.ix_(parents_idx,parents_idx)], self.prob['pot_vec'][i])

		prob_pad = {'mean': mean, 'var': var}

		return prob_pad	

