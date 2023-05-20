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


class ucb_acq(acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N, beta=1):
		super().__init__(sigma_square, mean, var, mu_target, N)
		self.beta = beta

		nnodes = len(mean)
		self.nnodes = nnodes
		self.est_a = np.matmul(np.eye(nnodes) - mean, mu_target)

		A = inv(np.eye(nnodes) - mean)
		self.var_mu = np.matmul(var, mu_target)
		self.mu_var_mu = np.matmul(mu_target.T, self.var_mu).reshape(-1,1)

	def evaluate(self, a):
		a = a.reshape(-1, 1)

		d1 = np.linalg.norm(a - self.est_a, ord=2)**2
		d2 = self.__stats(a)
		return d1 - self.beta*d2

	def __stats(self, a, grad=False):
		d1 = (a - self.est_a)**2
		d = np.sum((self.sigma_square).reshape(-1,1) * (d1*self.mu_var_mu))
		return max(d,0)**0.5


class sample_acquisition():
	def __init__(self, sigma_square, mean, var, mu_target, n):
		self.sigma_square = sigma_square.reshape(-1,1)
		self.mean = mean
		self.var = var
		self.mu_target = mu_target
		self.n = n

	def sample_B(self):
		B = []
		for i in range(len(self.mean)):
			B.append(np.random.multivariate_normal(self.mean[i], self.var[i]))
		B = np.array(B)	
		return B

	def sample_x(self, a):
		B = self.sample_B()
		eps = np.random.multivariate_normal(np.zeros(len(a.flatten())), np.diag(self.sigma_square.flatten()))
		return np.linalg.inv(np.eye(len(a.flatten()))-B) @ (a+eps)
		
	def optimize(self, x0, num_samples=10):

		fmin = self.evaluate(x0)
		amin = x0

		for _ in range(num_samples):
			a_jitter = x0.reshape(-1)*np.random.uniform(0.8,1.2,(len(x0.flatten()),))
			x01 = np.maximum(np.minimum(a_jitter, 1), -1)
			x01 = x01 / np.linalg.norm(x01)
			fvalue = self.evaluate(x01)
			if fvalue < fmin:
				fmin = fvalue
				amin = x01

		return amin

	def evaluate(self, a, mc_samples=200):
		raise NotImplementedError
	

class ei_acq(sample_acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)

		nnodes = len(mean)
		self.nnodes = nnodes
		self.obj_min = np.inf

	def evaluate(self, a, mc_samples=10):
		a = a.reshape(-1, 1)
		sum = 0
		for _ in range(mc_samples):
			B = self.sample_B()
			obj_a = np.linalg.norm((np.linalg.inv(np.eye(self.nnodes)-B))@a - self.mu_target)**2
			sum += min(obj_a, self.obj_min)
		return sum/mc_samples
	

class mi_acq(sample_acquisition):
	def __init__(self, sigma_square, mean, var, mu_target, N):
		super().__init__(sigma_square, mean, var, mu_target, N)


	def evaluate(self, a, mc_samples=10):
		a = a.reshape(-1,1)

		sum = 0
		for _ in range(mc_samples):
			x = self.sample_x(a)

			d2 = 1/self.n + np.matmul(x.T, np.matmul(self.var, x)) 	
			var_scaled_a = np.matmul(self.var, x)
			var_a = self.var - np.matmul(var_scaled_a, np.transpose(var_scaled_a, (0,2,1))) / d2

			mu_target = self.mu_target.reshape(-1,1)
			log_vars = [np.log(max(mu_target.T@var_a[i]@mu_target, 1e-6)).item() for i in range(len(var_a))]
			sum += np.sum(log_vars).item()

		return sum / mc_samples