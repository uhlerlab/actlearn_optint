import numpy as np
from numpy.linalg import inv
from scipy.optimize import minimize,Bounds


class acquisition_known_dag():
	def __init__(self, sigma_square, mean, var, mu_target, N):
		self.sigma_square = sigma_square.reshape(-1,1) # (p,1)
		self.mean = mean # (p,p)
		self.var = var # (p,p,p)
		self.mu_target = mu_target # (p,1)
		self.N = N

	def optimize(self, a_pool):
		return min(a_pool, key = lambda a: self.evaluate(a))

	def evaluate(self, a):
		raise NotImplementedError


class ivr_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N, method):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)

		k = np.matmul(mean, mu_target) - mu_target # (p,1)
		if method == 'sphere':
			self.c0 = k**2 + 1/nnodes
		elif method == 'cube':
			self.c0 = k**2 + 1/3
		elif method == 'iw_approx1':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.weight = self.weight / np.sum(self.weight)
			self.c0 = 8 * self.weight
		elif method == 'iw_approx2':
			self.weight = k**2 / np.sum(k**2)
			self.weight = (self.weight + 1/(2*nnodes)) / np.sum(self.weight + 1/(2*nnodes))
			self.c0 = 8 * self.weight
		elif method == 'iw_approx1+':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.weight = self.weight / np.sum(self.weight)
			self.c0 = 16 * self.weight
		else:
			assert False, f"Unsupported weighting method for ivr: {method}"

		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)


	def evaluate(self, a):
		a = a.reshape(-1,1)

		if np.linalg.norm(self.sigma_square) != 0:
			d = self.sigma_square * self.__stats(a)
			return np.dot(d.T, d + 2 * self.sigma_square + 2 * self.c0).item()
		else:
			d = self.__stats(a)
			return np.dot(d.T, 2 * self.c0).item()

	def __stats(self, a):
		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)

		d = (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)
		return d


# class svr_known_dag(acquisition_known_dag):
# 	def __init__(self, sigma_square, mean, var, mu_target, N, a_pool):
# 		super().__init__(sigma_square, mean, var, mu_target, N)

# 		nnodes = len(mean)
# 		A = inv(np.eye(nnodes) - mean) # (p,p)
# 		self.var_scaled = np.matmul(var, A) # (p,p,p)
# 		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
# 		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)

# 		# b = np.matmul((np.eye(nnodes) - mean), mu_target)
# 		b = min(a_pool, 
# 			key = lambda a: np.linalg.norm(a - np.matmul(np.eye(nnodes)-mean, mu_target))
# 			)
# 		self.ab_square = ((a_pool - b)**2).mean(axis=1).reshape(-1,1)

# 	def evaluate(self, a):
# 		a = a.reshape(-1,1)

# 		v = self.sigma_square * self.__calv(a)
		
# 		b0 = np.dot(v.T, v + 2 * self.sigma_square)
# 		b1 = 2 * np.dot(v.T, self.ab_square)
# 		return b0 + b1

# 	def __calv(self, a):
# 		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
# 		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)

# 		d = (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)
# 		return d


class mestvr_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)

	def evaluate(self, a):
		a = a.reshape(-1,1)

		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) 	
		var_scaled_a = np.matmul(self.var_scaled, a) # (p,p,1)
		var_a = self.var - np.matmul(var_scaled_a, np.transpose(var_scaled_a, (0,2,1))) / d2 # (p,p,p)

		return np.linalg.norm(var_a, ord=2, axis=(1,2), keepdims=True).max()


class svr_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)

	def evaluate(self, a):
		a = a.reshape(-1, 1)

		d= self.__stats(a)
		return d.max()

	def __stats(self, a, grad=False):
		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)	
		return d	


class ml2_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)
		
		nnodes = len(mean)
		self.est_a = np.matmul(np.eye(nnodes) - mean, mu_target)

	def evaluate(self, a):
		a = a.reshape(-1,1)
		return np.linalg.norm(a - self.est_a, ord=2)