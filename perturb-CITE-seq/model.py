import numpy as np
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression


class bayes_estimate_given_DAG():

	def __init__(self, DAG, hparams, preprocess=None):
		self.DAG = DAG
		self.p = DAG.nnodes
		self.hparams = hparams

		self.prob = self._prior()

		assert preprocess in {None, 'nonzero'}, "Preprocessing method not supported!"
		self.preprocess = preprocess

		self.samples = 0

	def _prior(self):
		pot_vec = [self.hparams['pot_vec'] * np.ones(self.DAG.indegree_of(i)) for i in range(self.p)]
		info_mat = [self.hparams['info_mat'] * np.eye(self.DAG.indegree_of(i)) for i in range(self.p)]
		sigma_square = np.zeros(self.DAG.nnodes)

		self.prob = {'pot_vec': pot_vec, 'info_mat': info_mat, 'sigma_square':sigma_square}
		return self.prob

	def update_posterior(self, a, batch):	
		batch_size = batch.shape[1]
	
		for i in range(self.p):
			parents_idx = list(self.DAG.parents_of(i))
			parents_idx.sort()

			X_par = batch[parents_idx, :]
			X_i_correct = batch[i, :] - a[i]
		
			if self.preprocess == 'nonzero':
				sample_idx = np.where(np.prod(batch[parents_idx+[i],:], axis=0)>0)[0]
				if len(sample_idx) != 0:
					X_par = X_par[:, sample_idx]
					X_i_correct = X_i_correct[sample_idx]

			self.prob['info_mat'][i] += np.matmul(X_par, X_par.T)
			self.prob['pot_vec'][i] += np.matmul(X_i_correct, X_par.T)

			if np.linalg.norm(a) == 0:
				if len(parents_idx) == 0:
					batch_var = np.var(X_i_correct)
				else:
					batch_var = np.var(LinearRegression().fit(X_par.T, X_i_correct).predict(X_par.T) - X_i_correct)
				self.prob['sigma_square'][i] += (batch_var - self.prob['sigma_square'][i]) #* batch.shape[1] / (self.samples)

		self.samples += batch_size


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