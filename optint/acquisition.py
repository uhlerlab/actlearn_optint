import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize, Bounds


class acquisition():
	def __init__(self, sigma_square, mean, var, mu_target, n):
		self.sigma_square = sigma_square.reshape(-1,1)
		self.mean = mean
		self.var = var
		self.mu_target = mu_target
		self.n = n

	def optimize(self, x0, grad=None):
		res = minimize(
			self.evaluate, 
			x0=x0, 
			method='SLSQP',
			jac=grad, 
			bounds=Bounds(-1,1), 
			constraints={'type':'ineq', 'fun': lambda x: 1-np.linalg.norm(x), 'keep_feasible': True}
			)	# keep the feasible set l2-norm <= 1
		if res.success:
			return res.x 
		else:
			print("Optimization fails...")
			print(res.message)
			return None

	def grad(self, a):
		# impute gradient formula
		pass

	def evaluate(self, a):
		raise NotImplementedError


class civ_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, n, measure):
		super().__init__(sigma_square, mean, var, mu_target, n)

		nnodes = len(mean)

		k = np.matmul(mean, mu_target) - mu_target # (p,1)
		if measure == 'unif':
			self.c1 = self.sigma_square + k**2 + 1/nnodes
		elif measure == 'ow':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.c1 = self.sigma_square + 16 * self.weight / np.sum(self.weight)
		else:
			assert False, "Unsupported measure for civ: {}".format(measure)

		A = inv(np.eye(nnodes) - mean)
		self.var_scaled = np.matmul(var, A) 
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) 


	def evaluate(self, a):
		a = a.reshape(-1,1)

		d= self.__stats(a)
		return np.dot(d.T, d + 2 * self.c1).item()

	def __stats(self, a, grad=False):
		# return the changed variance due to new point
		d1 = np.matmul(self.mu_var_scaled, a)
		d2 = 1/self.n + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a))

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


class maxv_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean)
		self.var_scaled = np.matmul(var, A)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled)

	def evaluate(self, a):
		a = a.reshape(-1,1)

		d2 = 1/self.n + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) 	
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
		d2 = 1/self.n + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a))

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1)
		return d	

