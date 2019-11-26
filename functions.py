
import numpy as np
import scipy

logit = lambda x: 1./ (1 +np.exp(-x))

scale_factor_vec = np.array([1e-9, 1e-8, 1e-7, 5e-6, 1e-6, 5e-5, 1e-5, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5])


def add_noise(params, sd=0.1):
    K = params.shape[0]
    noise= np.random.normal(0, sd, K)
    params = params + noise
    return params


def scale_factor_warm_up(x, y, noise_sigma,  prior_mean, prior_sigma, mode='rms', warmup_iters=150, batch_size=100, scale_factor_vec= scale_factor_vec):
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
			grad_vec = elbo_grad_gaussian(zs, means, betas, X_batch, Y_batch, noise_sigma, prior_mean, prior_sigma)*N/batch_size
			mean_grads = grad_vec[:K]
			betas_grad= grad_vec[K:]

			if i < 1:
				s[i] = 0.1*grad_vec**2
			else:

				s[i] = 0.1*grad_vec**2 + 0.9*s[i-1]
			if i//50 == 1:
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

		elbo_scale[j] = elbo_full(1000, means, betas, x, y, noise_sigma)
	#print(elbo_scale)
	#exit()
	#print(elbo_scale)
	print(np.nanargmax(elbo_scale))
	print(elbo_scale)
	scale_factor = scale_factor_vec[np.nanargmax(elbo_scale)]
	return scale_factor


def reparametrize(zs, means, log_sigmas):
	samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
	return samples

def elbo_grad_gaussian(zs, means, betas, x, y, noise_sigma, prior_mean, prior_sigma):
	K = x.shape[1]
	N = x.shape[0]
	noise_var = noise_sigma**2
	#print(noise_sigma)
	mean_grads = np.zeros((K,1))
	beta_grads = np.zeros((K,1))
	samples = reparametrize(zs, means, betas)
	zeta = x @ samples
	mean_grads = x.T@(y - zeta)/noise_var
	print(mean_grads)
	mean_grads = mean_grads + log_prior_grad(zs, prior_mean, prior_sigma)
	print(mean_grads)
	sigmas = np.exp(betas)
	beta_grads = mean_grads*zs*sigmas[:,None] + 1.
	grad_vec=  np.concatenate((mean_grads.flatten(), beta_grads.flatten()), axis=0)
	return grad_vec

def log_prior_grad(zs, prior_mean, prior_sigma):
	prior_var = prior_sigma
	#print(zs.shape)
	#print(prior_mean.shape)
	#print(prior_var.shape)
	prior_grad = -(zs-prior_mean[:,np.newaxis]) /(2*prior_var[:,np.newaxis])
	#print(prior_grad.shape)
	return  prior_grad


def logp_data_gaussian(zs, means, betas, x, y, noise_sigma):
	sigmas = np.exp(betas)
	samples = reparametrize(zs, means, betas)
	zeta = x @ samples
	#print(x.shape)
	#print(y.shape)
	#print(zeta.shape)
	#import ipdb
	#log_density = np.sum(scipy.stats.multivariate_normal.logpdf(y, zeta.ravel(), noise_sigma))
	log_density = -y.size*np.log(2*np.pi) - np.sum((y-zeta)**2/(2*noise_sigma**2))
	print(log_density)
	return log_density

def gaussian_entropy(betas):
	return 1 + np.log(2*np.pi) + np.sum(betas)


def elbo_gaussian(zs, means, betas, x, y, noise_sigma):
	log_p = logp_data_gaussian(zs, means, betas, x,y, noise_sigma)
	elbo = np.mean(log_p) + gaussian_entropy(betas)
	return elbo


def elbo_full(n_samples, means, betas, x, y, noise_sigma):
	K = x.shape[1]
	elbo_samples = np.zeros((n_samples,1))
	#for i in np.arange(n_samples):
	zs = np.random.normal(0, 1, (K,1))
	elbo_samples = logp_data_gaussian(zs, means, betas, x, y, noise_sigma=noise_sigma)
	return np.mean(elbo_samples) + gaussian_entropy(betas)



def advi(X, Y, noise_sigma, n_elbo_samples= 2000, epochs = 10000, thinning=20, scale_factor=None, S=1000,
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

		grad_vec = elbo_grad_gaussian(zs, means, betas, X, Y, noise_sigma)
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
			test_lpd[i] = logp_data_gaussian(zs, means, betas, Xstar, Ystar)/Ystar.shape[0]

		if compute_K_hat:
			vi_samples = posterior_samples(means, betas, S)
			#log_proposal = log_proposal_density(vi_samples, means, betas)
			#log_target = log_target_density_logit(vi_samples, x, y)
			ip_ratio = log_importance_ratio(vi_samples, means, betas, X, Y, 1000)


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


def elbo_logit(zs, means, betas, x, y):
	log_p = logp_data_logit(zs, means, betas, x,y)
	elbo= np.mean(log_p) + gaussian_entropy(betas)
	return elbo

def elbo_full_logit(n_samples, means, betas, x, y):
	K = x.shape[1]
	elbo_samples = np.zeros((n_samples,1))
	#for i in np.arange(n_samples):
	zs = np.random.normal(0, 1, (K,))
	elbo_samples = logp_data_logit(zs, means, betas, x, y)
	return np.mean(elbo_samples) + gaussian_entropy(betas)


def scale_factor_warmup_logit(x, y, mode='rms', warmup_iters=150, batch_size=100, scale_factor_vec=scale_factor_vec):
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

		elbo_scale[j] = elbo_full_logit(1000, means, betas, x, y)
	#print(np.nanargmax(elbo_scale))
	scale_factor = scale_factor_vec[np.nanargmax(elbo_scale)]
	print(elbo_scale)
	#print(scale_factor)
	#print(scale_factor_vec)
	#print(np.nanargmax(elbo_scale))
	#exit()
	return scale_factor


def advi_logit(X, Y, n_elbo_samples= 2000, epochs = 10000, thinning=20, scale_factor=None, S=1000,
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
	logq = scipy.stats.norm.pdf(vi_samples, means, sigmas)
	logq_sum = np.sum(np.log(logq), axis=1)
	return logq_sum


def log_target_density_logit(vi_samples, x, y):
	t = vi_samples.T @ x
	p = logit(t)
	lp = np.mean(np.log(p) * y + np.log(1. - p) * (1. - y), axis=1)
	lp_full = np.sum(lp, axis=0)
	return lp_full
