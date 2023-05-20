import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt



def plot_mean(problem, opts, R, best=False, miscolor=False, **kwargs):
	plt.clf()

	fig, axs = plt.subplots(1,1, figsize=(5.4,3.7))
	plt.rcParams.update({'font.size': 14})

	exps = [k for k in kwargs.keys()]
	MSEs = {k:[] for k in exps}

	for r in range(R):
		for k in exps:
			mse = []
			for prob in kwargs[k][r]:
				a = min(
					problem.a_pool, 
					key = lambda a: np.linalg.norm(a-np.matmul(np.eye(problem.nnodes) - prob['mean'], problem.mu_target))
					)
				for key in problem.int_pool.keys():
					if max(abs(problem.int_pool[key][0]-a.reshape(-1))) == 0:
						mu_a = problem.int_pool[key][2].reshape(-1,1)
						errs = problem.mu_target - (mu_a-problem.mu)
						mse.append(np.linalg.norm(errs)**2)
						break	
				if best:
					mse[-1] = min(mse)
			MSEs[k].append(mse)

	if len(exps) == 7:
		colors = ['#069AF3', '#9ACD32', 'black', 'grey', 'orange', '#C79FEF', 'coral']
		markers = ['s', 'o', 'o', 'o', '^', '^', '^']		
	elif len(exps) == 6:
		colors = ['#069AF3', '#9ACD32', 'black', 'grey', 'orange', '#C79FEF'] 
		markers = ['s', 'o', 'o', 'o', '^', '^']
	elif len(exps) == 3:
		if miscolor:
			colors = ['#069AF3', 'orange', 'coral']
			markers = ['s', '^', '^']
		else:
			colors = ['#069AF3', '#9ACD32', 'orange']
			markers = ['s', 'o', '^']
	elif len(exps) == 4:
		colors = ['#069AF3', '#9ACD32', 'orange', 'red']
		markers = ['s', 'o', '^', '^']		
	for i,k in enumerate(exps):
		axs.plot(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0),
			label=k,
			linewidth=2,
			marker=markers[i],
			markersize=6,
			markeredgewidth=2,
			color=colors[i]
		)
		axs.fill_between(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0) - 0.1*np.array(MSEs[k]).std(axis=0),
			np.array(MSEs[k]).mean(axis=0) + 0.1*np.array(MSEs[k]).std(axis=0),
			alpha=.2,
			color=colors[i]
		)

	axs.spines.right.set_visible(False)
	axs.spines.top.set_visible(False)
	axs.set_xlabel(r'time step $t$')
	axs.set_ylabel(r'$||\mu_t^*-\mu^*||_2^2$')
	axs.legend(loc='upper right', fontsize=12)

	fig.tight_layout()

