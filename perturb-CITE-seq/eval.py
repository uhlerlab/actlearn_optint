import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt



def plot_mean(problem, opts, R, best=False, **kwargs):
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

	if len(exps) == 6:
		colors = ['#069AF3', '#9ACD32', 'black', 'grey', 'orange', '#C79FEF'] 
		markers = ['s', 'o', 'o', 'o', '^', '^']
	elif len(exps) == 3:
		colors = ['#069AF3', '#9ACD32', 'orange']
		markers = ['s', 'o', '^']
	for i,k in enumerate(exps):
		axs.plot(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0),
			label=k,
			linewidth=2,
			marker=markers[i],
			# markerfacecolor='white',
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

	# axs.set_xlim(-0.5, 9.5)
	# axs.set_xticks([0,2,4,6,8])
	# axs.set_yticks([2.1,2.2,2.3,2.4])
	# if best:
	# 	fig.suptitle('Best Mean')
	# else:
	# 	fig.suptitle('Mean')	
	# fig.suptitle('square distance to target mean')
	fig.tight_layout()

	# plt.show()


def plot_hit(problem, opts, R, **kwargs):
	plt.clf()

	fig, axs = plt.subplots(1,1, figsize=(6,4))

	exps = [k for k in kwargs.keys()]
	HITs = {k:[] for k in exps}

	for r in range(R):
		for k in exps:
			hit = [0]
			for prob in kwargs[k][r]:
				# a = min(
				# 	problem.a_pool, 
				# 	key = lambda a: np.linalg.norm(a-np.matmul(np.eye(problem.nnodes) - prob['mean'], problem.mu_target))
				# 	)
				i = np.argmin(np.matmul(np.eye(problem.nnodes) - prob['mean'], problem.mu_target))
				for b in problem.a_pool:
					if b[i] != 0:
						a = b
						break
				errs = abs(a - problem.a_target)
				if max(errs) == 0:
					hit.append(hit[-1]+1)
				else:
					hit.append(hit[-1])
			HITs[k].append(hit[1:])

	l = len(kwargs)
	colors = ['orange', '#9ACD32', '#069AF3'] + ['k'] * (l-3)
	markers = ['^', 'o', 's'] + ['s'] * (l-3)	
	for i,k in enumerate(exps):
		axs.plot(
			range(opts.T),
			np.array(HITs[k]).mean(axis=0),
			label=k,
			color=colors[i],
			marker=markers[i],
			markersize=6,
			markeredgewidth=2
		)
		axs.fill_between(
			range(opts.T),
			np.array(HITs[k]).mean(axis=0) - 0.1*np.array(HITs[k]).std(axis=0),
			np.array(HITs[k]).mean(axis=0) + 0.1*np.array(HITs[k]).std(axis=0),
			alpha=.2,
			color=colors[i]
		)

	axs.set_xlabel('Round')
	axs.set_ylabel('Hits')
	axs.legend(loc='upper right')


	fig.suptitle('Hits')	
	fig.tight_layout()


def plot_guess(problem, opts, R, best=False, **kwargs):
	plt.clf()

	fig, axs = plt.subplots(1,1, figsize=(4.8,3.5))

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
				errs = abs(a - problem.a_target)
				mse.append(np.concatenate(errs).mean())
				# mse.append(np.linalg.norm(errs))
				if best:
					mse[-1] = min(mse)
			MSEs[k].append(mse)

	l = len(kwargs)
	colors = ['orange', '#9ACD32', '#069AF3'] + ['k'] * (l-3)
	markers = ['^', 'o', 's'] + ['s'] * (l-3)
	for i, k in enumerate(exps):
		axs.plot(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0),
			label=k,
			linewidth=2.5,
			marker=markers[i],
			# markerfacecolor='white',
			markersize=6,
			markeredgewidth=2,
			color=colors[i]
		)
		axs.fill_between(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0) - 0.2*np.array(MSEs[k]).std(axis=0),
			np.array(MSEs[k]).mean(axis=0) + 0.2*np.array(MSEs[k]).std(axis=0),
			alpha=.2,
			color=colors[i]
		)

	axs.set_xlabel(r'time steps $t$')
	axs.set_ylabel(r'$|a_t^*-a^*|$')
	axs.legend(loc='upper right')

	# axs.set_xlim(-0.5,9.5)
	# axs.set_xticks([0,2,4,6,8])
	# axs.set_yticks([0.147, 0.151, 0.155, 0.159])
	

	# if best:
	# 	fig.suptitle('Best Matching Intervention')
	# else:
	# 	fig.suptitle('Matching Intervention')	
	fig.suptitle('mean absolute error of learned interventions')
	fig.tight_layout()

	# plt.show()



def plot_a(problem, opts, R, best=False, **kwargs):
	plt.clf()

	fig, axs = plt.subplots(1,1, figsize=(6,4))

	exps = [k for k in kwargs.keys()]
	MSEs = {k:[] for k in exps}

	for r in range(R):
		for k in exps:
			mse = []
			for a in kwargs[k][r]:
				errs = abs(a - problem.a_target)
				mse.append(np.concatenate(errs).mean())
				if best:
					mse[-1] = min(mse)
			MSEs[k].append(mse)

	for k in exps:
		axs.plot(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0),
			label=k	
		)
		axs.fill_between(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0) - 0.1*np.array(MSEs[k]).std(axis=0),
			np.array(MSEs[k]).mean(axis=0) + 0.1*np.array(MSEs[k]).std(axis=0),
			alpha=.2
		)

	axs.set_xlabel('Round')
	axs.set_ylabel('Mean Absolute Error')
	axs.legend(loc='upper right')

	if best:
		fig.suptitle('Best Picked Intervention')
	else:
		fig.suptitle('Picked Intervention')
	fig.tight_layout()

	plt.show()



def plot_a_hit(problem, opts, R, **kwargs):
	plt.clf()

	fig, axs = plt.subplots(1,1, figsize=(4.8,3.5))

	exps = [k for k in kwargs.keys()]
	HITs = {k:[] for k in exps}

	for r in range(R):
		for k in exps:
			hit = [0]
			for a in kwargs[k][r]:
				errs = abs(a - problem.a_target)
				if max(errs) == 0:
					hit.append(hit[-1]+1)
				else:
					hit.append(hit[-1])
			HITs[k].append(hit[1:])

	colors = ['orange', '#9ACD32', '#069AF3', 'k']
	markers = ['^', 'o', 's', 's']
	for i,k in enumerate(exps):
		axs.plot(
			range(opts.T),
			np.array(HITs[k]).mean(axis=0),
			label=k,
			linewidth=2.5,
			marker=markers[i],
			# markerfacecolor='white',
			markersize=6,
			markeredgewidth=2,
			color=colors[i]
		)
		axs.fill_between(
			range(opts.T),
			np.array(HITs[k]).mean(axis=0) - 0.2*np.array(HITs[k]).std(axis=0),
			np.array(HITs[k]).mean(axis=0) + 0.2*np.array(HITs[k]).std(axis=0),
			alpha=.2,
			color=colors[i]
		)

	axs.set_xlabel(r'time step $t$')
	axs.set_ylabel(r'number of $a_s=a^*$ ($s\leq t$)')
	axs.legend(loc='upper left')

	# axs.set_xlim(-0.5,9.5)
	# axs.set_xticks([0,2,4,6,8])
	# axs.set_yticks([0,1,2,3])

	fig.suptitle('accumulate hit times')	
	fig.tight_layout()

	# plt.show()