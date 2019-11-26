import numpy as np
from helper import compute_l2_norm
from helper import expectation_iw, gaussian_entropy
from scipy.stats import stats


def stepsize_cyclical_adaptive_schedule(lr, epoch_current, epochs, c):
	# this is the schedule given in the blog for SWA
	step_size = 0.01
	decay_rate = 0.9
	lr_multiplier = 2.5

	if epoch_current / epochs < 0.5:
		step_size = lr
	elif epoch_current/epochs > 0.5 and epoch_current/epochs < 0.75:
		step_size = lr * decay_rate
	else:
		step_size = cyclical_step_size_schedule(lr, 2.5*lr, epoch_current, c)

	return step_size


def stepsize_linear_adaptive_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0, weight=1.):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	weight= 1.5

	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.
		if params_swa is not None:
			means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
		else:
			means_swa = params[0]
			logsigmas_swa = params[1]
		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		# print(params)
		# print(params_swa)
		# print(epoch_current)
		# print(swa_start)
		# increase the current reduced learning rate in order to explore more ...
		#step_size = step_size*lr_multiplier
		step_size = lr_max
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n


def stepsize_linear_adaptive_is_mixing_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, densities, params_swa=None, swa_n=0, weight=1.):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	weight= 4.5
	debug = False
	new_densities = dict()
	new_densities['sd'] =densities['sd']
	new_densities['sum'] = densities['sum']
	new_densities = densities.copy()

	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.
		sample_density = densities['sd']
		sample_density_sum = densities['sum']

		if params_swa is not None:
			print('swa computations')
			print(sample_density_sum)
			print(sample_density)

			means_swa = [(i*sample_density_sum + j*sample_density)/(sample_density_sum + sample_density) for i,j in zip(params_swa[0], params[0])]
			#logsigmas_swa = [np.log((np.exp(i)*sample_density_sum + np.exp(j)*sample_density)/(sample_density_sum + sample_density)) for i,j in zip(params_swa[1], params[1])]
			logsigmas_swa = [(i*sample_density_sum + j*sample_density)/(sample_density_sum + sample_density) for i,j in zip(params_swa[1], params[1])]


			if debug:
				print(params_swa[0])
				print(params[0])
				print(params[1])
				print(params_swa[1])
				print(np.array(means_swa))
				print(np.array(logsigmas_swa))
				print(sample_density_sum)
				print(sample_density)

#			if swa_n == 1:
#				exit()


			#means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			#logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
			sample_density_sum += sample_density
			#mean_density_sum += mean_density
			new_densities['sum'] = sample_density_sum
		else:
			means_swa = params[0]
			logsigmas_swa = params[1]
			sample_density_sum = sample_density

		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		# print(params)
		# print(params_swa)
		# print(epoch_current)
		# print(swa_start)
		# increase the current reduced learning rate in order to explore more ...
		#step_size = step_size*lr_multiplier
		step_size = lr_max
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n, new_densities



def stepsize_adaptive_swa_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	if params_swa is None:
		params_swa = params.copy()

	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		step_size = lr_max
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 1:
		if params_swa is not None:
			means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
		else:
			means_swa = params_swa[0]
			logsigmas_swa = params_swa[1]
		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n



def stepsize_linear_weight_averaging(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0, weight=1, pmz='std'):
	# weight averaging from the beginning
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	#weight = 3.
	#weight= 0.1
	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.

		if params_swa is not None:
			if pmz == 'log':
				means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
				logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
			else:
				means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
				logsigmas_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[1], params[1])]

		else:
			means_swa = params_swa[0]
			logsigmas_swa = params_swa[1]
		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		step_size = lr_max
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n


def rms_prop_gradient(epoch_current, mean_grads, sigma_grads, s_previous=None, eta=0.1):
	grad_vec= np.concatenate((mean_grads, sigma_grads), axis=0)
	if epoch_current <= 1:
		s = 0.5*grad_vec**2
	else:
		s= 0.2*grad_vec**2 + 0.7*s_previous

	rho = epoch_current**(-0.5 + 1e-12)*eta/(1. +np.sqrt(s))
	return rho, s


def step_size_rms_prop_schedule(params_rms, epoch_current, mean_grads, sigma_grads, s_previous=None, eta=0.1):
	grad_vec= np.concatenate((mean_grads.flatten(), sigma_grads.flatten()), axis=0)

	#print(grad_vec)

	K= mean_grads.size
	if epoch_current <= 1:
		s = 0.5*grad_vec**2
	else:
		s= 0.2*grad_vec**2 + (0.8)*s_previous

	means, betas  = params_rms[0], params_rms[1]
	rho = epoch_current**(-0.2 + 1e-16)*eta /(1. +np.sqrt(s))
	means = means + rho[:K]*mean_grads
	betas = betas + rho[K:]*sigma_grads


	if epoch_current % 600 == 0:
		print(epoch_current)
		print(means)
		print(betas)
		print('sigma diff:')
		print(rho[K:] *sigma_grads)
		#exit()
	#params_new = np.concatenate((means.flatten(), betas.flatten()), axis=0)
	params_new = [means, betas]
	return rho, s, params_new


def compute_moving_avg(params1, params2, alpha):
	params1 *= (1 - alpha)*params1
	params2 += params2*alpha
	return params1, params2


def cyclical_step_size_schedule(step_size_min, step_size_max, epoch_current, cycle_length):
	t = ((epoch_current)% cycle_length +1)/ cycle_length
	step_size_current = (1. - t)*step_size_max + t*step_size_min

	return step_size_current


scale_factor_vec = np.array([1e-8, 1e-7, 0.000001, 0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])


def scale_factor_warm_up(x, y, mode='rms', warmup_iters=150, batch_size=100, scale_factor_vec=scale_factor_vec):
	N = x.shape[0]
	K = x.shape[1]
	# K is the dimensionality of W in linear/logistic regression
	inds = np.random.randint(N, size=(batch_size,))
	N_scales= scale_factor_vec.size
	X_batch = x[inds,:]
	Y_batch = y[inds]
	elbo_scale = np.zeros((N_scales, 1))
	for j in np.arange(N_scales):
		scale = scale_factor_vec[j]
		epochs = warmup_iters
		means_iter = np.zeros((epochs+1, K))
		betas_iter = np.ones((epochs+1, K))
		rho = np.zeros((epochs, 2*K))
		s = np.zeros((epochs, 2*K))

		for i in np.arange(epochs):
			zs= np.random.normal(0,1, (K,1) )
			means = means_iter[i]
			betas = betas_iter[i]
			grad_vec = elbo_grad_logit(zs, means, betas, X_batch, Y_batch)*N/batch_size
			mean_grads = grad_vec[:K]
			betas_grad= grad_vec[K:]

			if i < 1:
				s[i] = 0.1*grad_vec**2
			else:

				s[i] = 0.1*grad_vec**2 + 0.9*s[i-1]
			if i//2 == 1:
				print(s[i])
				print(rho[i])

			rho[i]= (i+1)**(-0.2)*scale / (1. + np.sqrt(s[i]))
			if mode == 'rms':
				means = means + rho[i,:K] * mean_grads
				betas = betas + rho[i,K:] * betas_grad
			elif mode == 'clr':
				means = means + scale*mean_grads
				betas = betas + scale*betas_grad
			means_iter[i+1,:] = means
			betas_iter[i+1,:] = betas

		elbo_scale[j] = elbo_full(1000, means, betas, x, y)
	#print(np.nanargmax(elbo_scale))
	scale_factor = scale_factor_vec[np.nanargmax(elbo_scale)]
	print(elbo_scale)
	#print(scale_factor)
	#print(scale_factor_vec)
	#print(np.nanargmax(elbo_scale))
	#exit()
	return scale_factor

logit = lambda x: 1./ (1 +np.exp(-x))


def reparametrize(zs, means, log_sigmas):
	#print(zs.shape)
	#print(log_sigmas.shape)
	samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
	#print(samples.shape)
	return samples



def elbo_grad_logit(zs, means, betas, x, y):
	K = x.shape[1]
	N = x.shape[0]

	mean_grads = np.zeros((K,1))
	beta_grads = np.zeros((K,1))
	#zs = np.random.normal(0, 1, size=(K, 1))
	samples = reparametrize(zs, means, betas)
	#print(x.shape)
	#print(samples.shape)
	zeta = x @ samples
	p = logit(zeta)
	mean_grads = x.T@(y - p)
	sigmas = np.exp(betas)
	beta_grads = mean_grads*zs*sigmas[:,None] + 1.
	grad_vec=  np.concatenate((mean_grads.flatten(), beta_grads.flatten()), axis=0)
	return grad_vec


def logp_data_logit(zs, means, betas, x, y):
	sigmas = np.exp(betas)
	samples = reparametrize(zs, means, betas)
	zeta = x @ samples
	p = logit(zeta)
	log_density = np.sum(np.log(p) * y + np.log(1 - p + 1e-10) * (1. - y), axis=0)
	return log_density


def gaussian_entropy(betas):
	return 1 + np.log(2*np.pi) + np.sum(betas)


def elbo_logit(zs, means, betas, x, y):
	log_p = logp_data_logit(zs, means, betas, x,y)
	elbo= np.mean(log_p) + gaussian_entropy(betas)
	return elbo


def elbo_full(n_samples, means, betas, x, y):
	K = x.shape[1]
	elbo_samples = np.zeros((n_samples,1))
	#for i in np.arange(n_samples):
	zs = np.random.normal(0, 1, (K,))
	elbo_samples = logp_data_logit(zs, means, betas, x, y)
	return np.mean(elbo_samples) + gaussian_entropy(betas)


def advi(X, Y, n_elbo_samples= 2000, epochs = 10000, thinning=20, scale_factor=None, S=1000,
		 compute_elbo=True, test_elpd=True, compute_K_hat=True, Xstar=None, Ystar=None):
	scale_factor = scale_factor_warm_up(X, Y)
	K = X.shape[1]
	means_iter = np.zeros((epochs+1, K))
	betas_iter = np.ones((epochs+1, K))

	K_iter = np.zeros((epochs,1))
	first_moment_estimation = np.zeros((3, epochs, K))
	second_moment_estimation = np.zeros((3, epochs, K))


	elbo_iter = np.zeros((epochs,))
	rho = np.zeros((epochs, 2 * K))
	s = np.zeros((epochs, 2 * K))
	test_lpd = np.zeros((epochs,))
	elbo_iter_MC = np.array((epochs, 2*K))

	for i in np.range(epochs):
		zs = np.random.normal(0,1, (K,1))
		means = means_iter[i]
		betas = betas_iter[i]

		grad_vec = elbo_grad_logit(zs, means, betas, X, Y)
		mean_grads = grad_vec[:K]
		betas_grad = grad_vec[K:]
		if i == 1:
			s[i] = 0.1 * grad_vec ^ 2
		else:
			s[i] = 0.1 * grad_vec ^ 2 + 0.9 * s[i - 1]

		rho[i] = i ** (-0.2 + 1e-16) * scale_factor / (1. + np.sqrt(s[i]))

		means = means + rho[:K] * mean_grads
		betas = betas + rho[K:] * betas_grad
		means_iter[i + 1, :] = means
		betas_iter[i + 1, :] = betas

		if compute_elbo:
			elbo_iter[i]= elbo_full(100, means, betas, X, Y)


		if test_elpd and Ystar:
			test_lpd[i] = logp_data_logit(zs, means, betas, Xstar, Ystar)/Ystar.shape[0]

		if compute_K_hat:
			vi_samples = posterior_samples(means, betas, S)
			#log_proposal = log_proposal_density(vi_samples, means, betas)
			#log_target = log_target_density_logit(vi_samples, x, y)
			ip_ratio = log_importance_ratio(vi_samples, means, betas, X, Y, 1000)


def posterior_samples(means, vars, num_proposal_samples=1000):
	var_samples = np.random.multivariate_normal(means, np.diag(vars), size=num_proposal_samples)
	return var_samples


def log_importance_ratio(vi_samples, means, betas, X, Y, S=1000):
	if not vi_samples:
		vi_samples = posterior_samples(means, betas, num_proposal_samples=S)

	sigmas = np.exp(betas)
	if(vi_samples.shape[0] != S ):
		print("warning: mis-specify the length of samples")
	lp =log_proposal_density(vi_samples, means, betas)
	lt =log_target_density_logit(vi_samples, X, Y)
	return lt-lp


def log_proposal_density(vi_samples, means, betas):
	sigmas = np.exp(betas)
	logq = stats.norm.pdf(vi_samples, means, sigmas)
	logq_sum = np.sum(np.log(logq), axis=1)
	return logq_sum


def log_target_density_logit(vi_samples, x, y):
	t = vi_samples.T @ x
	p = logit(t)
	lp = np.mean(np.log(p) * y + np.log(1. - p) * (1. - y), axis=1)
	lp_full = np.sum(lp, axis=0)
	return lp_full



