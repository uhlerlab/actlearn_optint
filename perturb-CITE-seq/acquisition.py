import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize,Bounds


class acquisition():
	def __init__(self, sigma_square, mean, var, mu_target, N):
		self.sigma_square = sigma_square.reshape(-1,1)
		self.mean = mean
		self.var = var
		self.mu_target = mu_target
		self.N = N

	def optimize(self, a_pool):
		return min(a_pool, key = lambda a: self.evaluate(a))

	def evaluate(self, a):
		raise NotImplementedError


class civ_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N, measure):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)

		k = np.matmul(mean, mu_target) - mu_target
		if measure == 'unif':
			self.c0 = k**2 + 1/nnodes
		elif measure == 'ow':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.weight = self.weight / np.sum(self.weight)
			self.c0 = 16 * self.weight
		else:
			assert False, 'Unsupported weighting method for ivr: {}'.format(measure)

		A = inv(np.eye(nnodes) - mean)
		self.var_scaled = np.matmul(var, A) 
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled)


	def evaluate(self, a):
		a = a.reshape(-1,1)

		if np.linalg.norm(self.sigma_square) != 0:
			d = self.sigma_square * self.__stats(a)
			return np.dot(d.T, d + 2 * self.sigma_square + 2 * self.c0).item()
		else:
			d = self.__stats(a)
			return np.dot(d.T, 2 * self.c0).item()

	def __stats(self, a):
		d1 = np.matmul(self.mu_var_scaled, a)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) 
		d = (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1)
		return d


class maxv_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean)
		self.var_scaled = np.matmul(var, A)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled)

	def evaluate(self, a):
		a = a.reshape(-1,1)

		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) 	
		var_scaled_a = np.matmul(self.var_scaled, a)
		var_a = self.var - np.matmul(var_scaled_a, np.transpose(var_scaled_a, (0,2,1))) / d2

		return np.linalg.norm(var_a, ord=2, axis=(1,2), keepdims=True).max()


class cv_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean)
		self.var_scaled = np.matmul(var, A)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled)

	def evaluate(self, a):
		a = a.reshape(-1, 1)

		d= self.__stats(a)
		return d.max()

	def __stats(self, a, grad=False):
		d1 = np.matmul(self.mu_var_scaled, a)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a))

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1)	
		return d	


class greedy_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)
		
		nnodes = len(mean)
		self.est_a = np.matmul(np.eye(nnodes) - mean, mu_target)

	def evaluate(self, a):
		a = a.reshape(-1,1)
		return np.linalg.norm(a - self.est_a, ord=2)