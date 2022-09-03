import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


# def plot_matching_intervention(problems, )
def plot_acquisition(opts, acquisition):
	plt.plot(range(opts.T), acquisition, 'k')
	plt.title(opts.acq)
	plt.xlabel("Round")
	plt.ylabel("Acquisition value")
	plt.show()


def plot_shifted_mean(problem, opts, **kwargs):
	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (8,3)
	fig, axs = plt.subplots(1,2, figsize=figsize)

	exps = kwargs.keys()

	# last round statistics	
	Errs = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			errs = abs(np.dot(problem.A, np.dot(np.eye(problem.nnodes)-np.array(kwargs[k][r][-1]['mean']),problem.mu_target)))
			Errs[k].append(errs)

	axs[0].set_yscale('log')
	for k in exps:
		axs[0].errorbar(
			range(problem.nnodes), 
			np.array([np.concatenate(e) for e in Errs[k]]).mean(axis=0), 
			.1*np.array([np.concatenate(e) for e in Errs[k]]).std(axis=0), 
			label=k, 
			fmt='o'
			)
	axs[0].set_xlabel("Parameter Index")
	axs[0].set_ylabel("Absolute Error")
	axs[0].set_xticks(range(problem.nnodes))
	axs[0].grid(visible=True, axis='x')
	axs[0].set_title("Last round statistics")

	# all round statistics
	MSEs = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			mse = []
			for prob in kwargs[k][r]:
				errs = abs(np.dot(np.eye(problem.nnodes)-np.array(prob['mean']),problem.mu_target) - problem.a_target)
				mse.append(np.concatenate(errs).mean())
			MSEs[k].append(mse)

	axs[1].set_yscale('log')
	for k in exps:
		axs[1].errorbar(
			range(opts.T), 
			np.array(MSEs[k]).mean(axis=0), 
			.1*np.array(MSEs[k]).std(axis=0), 
			label=k
			)
	axs[1].set_xlabel('Round')
	axs[1].set_ylabel('Mean Absolute Error')
	axs[1].legend(loc='upper right')
	axs[1].set_title("All round statistics")

	fig.suptitle('Shifted Mean')
	fig.tight_layout()


def plot_matching_intervention(problem, opts, savefile=None, **kwargs):
	plt.clf()

	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (8,3)
	fig, axs = plt.subplots(1,2, figsize=figsize)

	exps = [k for k in kwargs.keys() if k!='figsize']

	# last round statistics
	Errs = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			errs = abs(np.dot(np.eye(problem.nnodes)-np.array(kwargs[k][r][-1]['mean']),problem.mu_target) - problem.a_target)
			Errs[k].append(errs)

	axs[0].set_yscale('log')
	for k in exps:
		axs[0].errorbar(
			range(problem.nnodes), 
			np.array([np.concatenate(e) for e in Errs[k]]).mean(axis=0), 
			.1*np.array([np.concatenate(e) for e in Errs[k]]).std(axis=0), 
			label=k, 
			fmt='o'
			)
	axs[0].set_xlabel("Parameter Index")
	axs[0].set_ylabel("Absolute Error")
	axs[0].set_xticks(range(problem.nnodes))
	axs[0].grid(visible=True, axis='x')
	axs[0].set_title("Last round statistics")

	MSEs = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			mse = []
			for prob in kwargs[k][r]:
				errs = abs(np.dot(np.eye(problem.nnodes)-np.array(prob['mean']),problem.mu_target) - problem.a_target)
				mse.append(np.concatenate(errs).mean())
			MSEs[k].append(mse)

	axs[1].set_yscale('log')
	for k in exps:
		axs[1].plot(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0),
			label=k	
		)
		axs[1].fill_between(
			range(opts.T),
			np.array(MSEs[k]).mean(axis=0) - 0.1*np.array(MSEs[k]).std(axis=0),
			np.array(MSEs[k]).mean(axis=0) + 0.1*np.array(MSEs[k]).std(axis=0),
			alpha=.2
		)
		# axs[1].errorbar(
		# 	range(opts.T), 
		# 	np.array(MSEs[k]).mean(axis=0), 
		# 	0.1*np.array(MSEs[k]).std(axis=0), 
		# 	label=k
		# 	)
		# print(np.array(MSEs[k]).mean(axis=0))
	axs[1].set_xlabel('Round')
	axs[1].set_ylabel('Mean Absolute Error')
	axs[1].legend(loc='upper right')
	axs[1].set_title("All round statistics")
	# axs[1].set_ylim(0.01,0.05)
	# axs[1].set_xlim(0,50)

	fig.suptitle('Matching Intervention')
	fig.tight_layout()

	if savefile is not None:
		plt.savefig(savefile)
	plt.show()
	plt.close()

	return MSEs


def plot_picked_intervention_j(problem, opts, A_active, j, **kwargs):
	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (8,2.2)
	fig, ax = plt.subplots(1,1,figsize=figsize)

	ax.imshow(np.concatenate(A_active[j], axis=1), norm=None, cmap='PiYG', vmin=-1, vmax=1)
	ax.set_xticks([2*i for i in range(opts.T//2)])
	ax.set_yticks([2*i for i in range(problem.nnodes//2)])
	ax.set_xlim([-.5,opts.T-.5])

	aset = set().union(*[problem.DAG.descendants_of(i) for i in range(problem.nnodes) if problem.a_target[i]!=0])
	for i in range(problem.nnodes):
		if problem.a_target[i]!=0:
			ax.arrow(opts.T+1.3,i,-.8,0,head_width = .6, width=.2, color='y', clip_on=False)
			ax.annotate("{:.2f}".format(problem.a_target[i][0]), (opts.T-1,i), (opts.T+1.5,i+0.2))
		elif i in aset:
			ax.arrow(opts.T+1.3,i,-.8,0,head_width = .6, width=.2, color='orange', clip_on=False)

	fig.suptitle(f'Picked Interventions ({j}th run)')

	fig.tight_layout()


def plot_picked_interventions(problem, opts, savefile=None, **kwargs):
	plt.clf()

	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (12, 3.2)
	
	exps = kwargs.keys()

	fig, ax = plt.subplots(len(exps),3, figsize=figsize)

	for i,k in enumerate(exps):
		im = ax[i][0].imshow(np.concatenate(kwargs[k][0], axis=1), norm=None, cmap='PiYG', vmin=-1, vmax=1)
		ax[i][0].set_xticks([2*i for i in range(opts.T//2)])
		ax[i][0].set_yticks([2*i for i in range(problem.nnodes//2)])
		ax[i][0].set_ylabel('Passive')

		ax[i][1].imshow(np.concatenate(kwargs[k][25], axis=1), norm=None, cmap='PiYG', vmin=-1, vmax=1)
		ax[i][1].set_xticks([2*i for i in range(opts.T//2)])
		ax[i][1].set_yticks([2*i for i in range(problem.nnodes//2)])

		ax[i][2].imshow(np.concatenate(kwargs[k][49], axis=1), norm=None, cmap='PiYG', vmin=-1, vmax=1)
		ax[i][2].set_xticks([2*i for i in range(opts.T//2)])
		ax[i][2].set_yticks([2*i for i in range(problem.nnodes//2)])

	cbaxes = fig.add_axes([1, 0.2, 0.01, 0.5]) 
	fig.colorbar(im, cax=cbaxes)

	fig.text(0.5, 0, 'Rounds')
	fig.text(0, 0.4, 'Dimension', rotation='vertical')

	fig.suptitle('Picked Interventions (3 runs)')

	fig.tight_layout()

	if savefile is not None:
		plt.savefig(savefile)
	plt.show()
	plt.close()


def plot_all_round(problem, opts, **kwargs):
	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (7, 3.3)
	fig, ax = plt.subplots(1,2,figsize=figsize)

	exps = kwargs.keys()

	MSEs = {k:[] for k in exps}
	LVars = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			mse = []
			lvar = []
			for prob in kwargs[k][r]:
				errs, vars = eval_B_rows(problem, prob)
				mse.append(np.concatenate(errs).mean())
				lvar.append(np.concatenate(vars).max())
			MSEs[k].append(mse)
			LVars[k].append(lvar)

	ax[0].set_yscale('log')
	for k in exps:
		ax[0].errorbar(range(opts.T), np.array(MSEs[k]).mean(axis=0), .1*np.array(MSEs[k]).std(axis=0))
	ax[0].set_xlabel('Round')
	ax[0].set_title('Mean Squared Error')

	ax[1].set_yscale('log')
	for k in exps:
		ax[1].errorbar(range(opts.T), np.array(LVars[k]).mean(axis=0), .1*np.array(LVars[k]).std(axis=0), label=k)
	ax[1].set_xlabel('Round')
	ax[1].set_title('Largest Variance')
	ax[1].legend(loc='upper right')

	fig.suptitle('All round statistics')
	fig.tight_layout()


def plot_last_round(problem, opts, **kwargs):
	if 'figsize' in kwargs.keys():
		figsize = kwargs['figsize']
	else:
		figsize = (12, 6)
	fig, axs = plt.subplots(2,3, figsize=figsize)

	exps = kwargs.keys()
	Errs = {k:[] for k in exps}
	Vars = {k:[] for k in exps}

	Errs_A = {k:[] for k in exps}

	for r in range(opts.R):
		for k in exps:
			errs, vars = eval_B_rows(problem, kwargs[k][r][-1])
			Errs[k].append(errs)
			Vars[k].append(vars)

			errs = eval_A_columns(problem, kwargs[k][r][-1])
			Errs_A[k].append(errs)

	# entry-wise
	axs[0][0].set_yscale('log')
	for k in exps:
		axs[0][0].errorbar(
			range(problem.DAG.num_arcs), 
			np.array([np.concatenate(e) for e in Errs[k]]).mean(axis=0), 
			.1*np.array([np.concatenate(e) for e in Errs[k]]).std(axis=0), 
			label=k
			)
	axs[0][0].set_title("Squared Error")

	axs[0][1].set_yscale('log')
	for k in exps:
		axs[0][1].errorbar(
			range(problem.DAG.num_arcs), np.array([np.concatenate(e) for e in Vars[k]]).mean(axis=0), 
			.5*np.array([np.concatenate(e) for e in Vars[k]]).std(axis=0), 
			label=k
			)
	axs[0][1].set_title("Variance")

	axs[0][2].set_yscale('log')
	for k in exps:
		axs[0][2].errorbar(
			range(problem.DAG.num_arcs), 
			np.array([np.concatenate(e) for e in Errs_A[k]]).mean(axis=0), 
			.1*np.array([np.concatenate(e) for e in Errs_A[k]]).std(axis=0), 
			label=k
			)
	axs[0][2].set_title("Squared Error")

	# dimension-wise
	axs[1][0].set_yscale('log')
	for k in exps:
		dim_eff = len([1 for f in Errs[k][0] if len(f)!=0]) 
		axs[1][0].errorbar(
			range(dim_eff), 
			np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Errs[k]]).mean(axis=0), 
			.1*np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Errs[k]]).std(axis=0), 
			label=k
			)
	axs[1][0].set_title("Squared Error")

	axs[1][1].set_yscale('log')
	for k in exps:
		axs[1][1].errorbar(
			range(dim_eff), 
			np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Vars[k]]).mean(axis=0), 
			.5*np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Vars[k]]).std(axis=0), 
			label=k
			)
	axs[1][1].set_title("Variance")

	axs[1][2].set_yscale('log')
	for k in exps:
		dim_eff = len([1 for f in Errs_A[k][0] if len(f)!=0]) 
		axs[1][2].errorbar(
			range(dim_eff), 
			np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Errs_A[k]]).mean(axis=0), 
			.1*np.array([np.array([f.mean() for f in e if len(f)!=0]) for e in Errs_A[k]]).std(axis=0), 
			label=k
			)
	axs[1][2].set_title("Squared Error")
	axs[1][2].legend(loc='upper right')

	fig.text(0.35, -0.01, r'$B$', fontsize=14)
	fig.text(0.85, -0.01, r'$(I-B)^{-1}$', fontsize=14)
	fig.text(-0.01, 0.2, 'Dimension', rotation='vertical', fontsize=14)
	fig.text(-0.01, 0.65, 'Parameter', rotation='vertical', fontsize=14)

	fig.suptitle('Last round statistics', fontsize=14)
	fig.tight_layout()


def eval_B_rows(problem, prob):
	errs = []
	vars = []

	for i in range(problem.nnodes):
		parents_idx = list(problem.DAG.parents_of(i))
		parents_idx.sort()
		
		err = (prob['mean'][i][parents_idx] - problem.B[i, parents_idx])**2
		var = prob['var'][i].diagonal()[parents_idx] * problem.sigma_square[i] 

		errs.append(err)
		vars.append(var)
	
	return errs, vars


def eval_B_columns(problem, prob):
	errs = []
	vars = []

	for i in range(problem.nnodes):
		children_idx = list(problem.DAG.children_of(i))
		children_idx.sort()

		err = (np.array([prob['mean'][j][i] for j in children_idx]) - problem.B[children_idx, i])**2
		var = np.array([prob['var'][j][i, i] * problem.sigma_square[j]  for j in children_idx])

		errs.append(err)
		vars.append(var)

	return errs, vars


def eval_A_columns(problem, prob):
	errs = []

	for i in range(problem.nnodes):
		children_idx = list(problem.DAG.children_of(i))
		children_idx.sort()

		bar_A = inv(np.eye(problem.nnodes) - np.array(prob['mean']))
		err = (bar_A[children_idx, i]- problem.B[children_idx, i])**2

		errs.append(err)

	return errs
