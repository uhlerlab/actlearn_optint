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

	def optimize(self, x0, grad=None):
		res = minimize(self.evaluate, x0=x0, method='SLSQP', jac=grad, bounds=Bounds(-1,1), constraints={'type':'ineq', 'fun': lambda x: 1-np.linalg.norm(x), 'keep_feasible': True}) 
		# keep the feasible set l2-norm <= 1
		if res.success:
			return res.x # (p,)
		# else:
			# res = minimize(self.evaluate, x0=x0, method='L-BFGS-B', jac=grad, bounds=Bounds(-1,1)) 
			# if res.success:
				# return res.x
		else:
			print("Optimization fails...")
			print(res.message)
			return None
		# 'TNC' can't handle constraints
		# res = minimize(self.evaluate, x0=x0, method='TNC', jac=grad, bounds=Bounds(-1,1))

	def grad(self, a):
		pass

	def evaluate(self, a):
		raise NotImplementedError


class ivr_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N, comp):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)

		k = np.matmul(mean, mu_target) - mu_target # (p,1)
		if comp == 'sphere':
			self.c1 = self.sigma_square + k**2 + 1/nnodes
		elif comp == 'cube':
			self.c1 = self.sigma_square + k**2 + 1/3
		elif comp == 'iw_approx1':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.c1 = self.sigma_square + 8 * self.weight / np.sum(self.weight)
		elif comp == 'iw_approx2':
			self.weight = k**2 / np.sum(k**2)
			self.weight = (self.weight + 1/(2*nnodes)) / np.sum(self.weight + 1/(2*nnodes))
			self.c1 = self.sigma_square + 8 * self.weight
		elif comp == 'iw_approx1+':
			self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
			self.c1 = self.sigma_square + 16 * self.weight / np.sum(self.weight)
		elif comp == 'iw_approx2+':
			self.weight = k**2 / np.sum(k**2)
			self.weight = (self.weight + 1/(2*nnodes)) / np.sum(self.weight + 1/(2*nnodes))
			self.c1 = self.sigma_square + 16 * self.weight	
		else:
			assert False, f"Unsupported comp for ivr: {comp}"

		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)

	# def grad(self, a):
	# 	# return gradient of the evaluation function
	# 	a = a.reshape(-1,1)
	# 	d, d1, d2 = self.__stats(a, grad=True)
	# 	mat2 = np.matmul(a.T, self.scaled_var_scaled) # (p,1,p)
	# 	mat = 2 * self.sigma_square.T * (-d1/d2 * self.mu_var_scaled + d1**2/d2**2 * mat2) # (p,1,p)
	# 	return 2 * np.matmul(np.squeeze(mat), d + self.c1).reshape(-1) # (p,1) 


	def evaluate(self, a):
		# return ivr up to linear transformation with positive first-order coefficient
		a = a.reshape(-1,1)

		d= self.__stats(a)
		return np.dot(d.T, d + 2 * self.c1).item()

	def __stats(self, a, grad=False):
		# return the changed variance due to new point
		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)
	
		# # normalize out the l2-norm effect of a
		# f = np.matmul(a.T, np.matmul(self.scaled_var_scaled, a))**0.5
		# d1 = d1 / np.maximum(f, 1e-3)
		# p,_,_ = d1.shape
		# d2 = (1/self.N + 1/self.sigma_square[0])*np.ones((p,1,1))

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)	
		# if grad:
		# 	return d, d1, d2
		return d

class ivr_test(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)

		k = np.matmul(mean, mu_target) - mu_target # (p,1)
		self.weight = (np.maximum(0.1,1 - k**2))**((4-nnodes)/2) * (k**2)
		self.c1 = self.weight / np.sum(self.weight)

		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)

	def evaluate(self, a):
		# return ivr up to linear transformation with positive first-order coefficient
		a = a.reshape(-1,1)

		d= self.__stats(a)
		return np.dot(d.T, self.c1).item()

	def __stats(self, a, grad=False):
		# return the changed variance due to new point
		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)
	
		# # normalize out the l2-norm effect of a
		# f = np.matmul(a.T, np.matmul(self.scaled_var_scaled, a))**0.5
		# d1 = d1 / np.maximum(f, 1e-3)
		# p,_,_ = d1.shape
		# d2 = (1/self.N + 1/self.sigma_square[0])*np.ones((p,1,1))

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)	
		# if grad:
		# 	return d, d1, d2
		return d

class ml2_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)
		
		nnodes = len(mean)
		self.est_a = np.matmul(np.eye(nnodes) - mean, mu_target)

	def evaluate(self, a):
		a = a.reshape(-1,1)
		return np.linalg.norm(a - self.est_a, ord=2)


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


class vr_known_dag(acquisition_known_dag):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		k = np.matmul(mean, mu_target) - mu_target # (p,1)
		self.c1 = self.sigma_square + k ** 2 # (p,1)

		nnodes = len(mean)
		A = inv(np.eye(nnodes) - mean) # (p,p)
		self.var_scaled = np.matmul(var, A) # (p,p,p)
		self.scaled_var_scaled = np.matmul(A.T, self.var_scaled) # (p,p,p)
		self.mu_var_scaled = np.matmul(mu_target.T, self.var_scaled) # (p,1,p)

	def evaluate(self, a):
		# return ivr up to linear transformation with positive first-order coefficient
		a = a.reshape(-1,1)

		d= self.__stats(a)
		return np.dot(d.T, d + 2 * self.c1).item()

	def __stats(self, a, grad=False):
		# return the changed variance due to new point
		d1 = np.matmul(self.mu_var_scaled, a) # (p,1,1)
		d2 = 1/self.N + np.matmul(a.T, np.matmul(self.scaled_var_scaled, a)) # (p,1,1)

		d = self.sigma_square * (np.matmul(self.mu_target.T, np.matmul(self.var, self.mu_target)) - d1**2/d2).reshape(-1,1) # (p,1)	
		# if grad:
		# 	return d, d1, d2
		return d