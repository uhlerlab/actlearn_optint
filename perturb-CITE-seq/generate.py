from graphical_models.classes.dags.dag import DAG
import numpy as np
from numpy.linalg import inv
import graphical_models as gm
import networkx as nx
import pickle
from copy import deepcopy


NNODES = 36
PATHDAG = "./data/estimated_dag.pkl"
PATHSAMPLES = "./data/data+.pkl" ######

class instance(object):

	def __init__(self, combination=False, seed=1234, target=None):
		print(PATHDAG)
		print(PATHSAMPLES)

		np.random.seed(seed)

		self.nnodes = NNODES
		
		with open(PATHDAG, 'rb') as file:
			dag = pickle.load(file)
			with open("./data/gene_dict.pkl", 'rb') as f:
				mapping = pickle.load(f)
				nx.relabel_nodes(dag, mapping, copy=False)
		self.DAG = DAG.from_nx(dag)
		
		with open(PATHSAMPLES, 'rb') as file:
			data = pickle.load(file)
			
		self.mu = data['ctrl'][2].reshape(-1,1)

		self.int_pool = {}
		self.ctrl_samples = None
		for k in data.keys():
			if k != 'ctrl':
				if combination:
					self.int_pool[k] = data[k]
				else:
					if len(k.split("_")) == 2:
						self.int_pool[k] = data[k]	
			else:
				self.ctrl_samples = data[k][3]
		self.a_pool = [self.int_pool[k][0].reshape(-1,1) for k in self.int_pool.keys()]
		self.a_pool_copy = deepcopy(self.a_pool)

		if target is not None:
			int_target = target
		else:
			int_target = np.random.choice(list(self.int_pool.keys()))
		self.mu_target = self.int_pool[int_target][2].reshape(-1,1) - self.mu
		self.a_target = self.int_pool[int_target][0].reshape(-1,1)
		

	# get N samples
	def sample(self, a, N=None):

		assert np.linalg.norm(a) == 0 or N, "Must input sample size for intervention!"

		if np.linalg.norm(a) == 0:
			if N is None:
				batch = self.ctrl_samples
			else:
				if self.ctrl_samples.shape[1] >= N:
					idx = np.random.choice(self.ctrl_samples.shape[1], N, replace=False)
				else:
					print("batch size exceeds sample size!")
					idx = np.arange(self.ctrl_samples.shape[1])
				batch = self.ctrl_samples[:, idx]
		else:
			for key in self.int_pool.keys():
				if np.linalg.norm(a-self.int_pool[key][0].reshape(-1,1)) == 0:
					k = key
					break
			all_batch = self.int_pool[k][3]

			if all_batch.shape[1] >= N:
				idx = np.random.choice(all_batch.shape[1], N, replace=False)
			else:
				print("batch size exceeds sample size!")
				idx = np.arange(all_batch.shape[1])
			batch = all_batch[:, idx]
		return batch
	
	def reduce_pool(self, a):
		ind = 0
		size = len(self.a_pool)
		while ind != size and not np.array_equal(self.a_pool[ind],a):
			ind += 1
		if ind != size:
			self.a_pool.pop(ind)
		else:
			raise ValueError('action not found in pool.')

	def restore_pool(self):
		self.a_pool = deepcopy(self.a_pool_copy)
