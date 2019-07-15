#!/usr/bin/env python
# coding: utf-8

import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import autograd.numpy as np
import seaborn as snb
import pystan

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule
from autograd import grad
np.set_printoptions(precision=3)


schools_code = """
data {
    int<lower=0> J; // number of schools
    vector[J] y; // estimated treatment effects
    vector<lower=0>[J] sigma; // s.e. of effect estimates
}
parameters {
    real mu;
    real<lower=0> tau;
    vector[J] eta;
}
transformed parameters {
    vector[J] theta;
    theta = mu + tau * eta;
}
model {
    eta ~ normal(0, 1);
    y ~ normal(theta, sigma);
}
"""

J = 8

schools_dat = {'J': J,
               'y': [28,  8, -3,  7, -1,  1, 18, 12],
               'sigma': [15, 10, 16, 11,  9, 11, 10, 18]}

sm = pystan.StanModel(model_code=schools_code)


# ### Run HMC
fit = sm.sampling(data=schools_dat, iter=10000, chains=2)

# ### Run Stan-ADVI
fit_vb = sm.vb(data=schools_dat, iter=100000, tol_rel_obj=1e-6, output_samples=5000)


# ### Run ADVI in Python

np.random.seed(123)

# Helper functions
def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(log_sigma):
    return np.sum(gaussian_entropy(log_sigma))

def reparametrize(zs, means, log_sigmas, lol):
    samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
    print('sigmas:')
    print(sigmas)
    print('log sigma grad:')
    log_sigma_grad = (sigmas[:, None]*zs).T
    print(log_sigma_grad)
    print(lol)
    return samples, log_sigma_grad

# use analytical gradient of entropy
compute_entropy_grad = grad(compute_entropy)

# settings
step_size = 1e-2
itt_max = 10000
num_samples = 1

# init variational params
num_params = 10
means = np.zeros(num_params)
sigmas = np.ones(num_params)
log_sigmas = np.log(sigmas)

params = [means, sigmas]
params_swa = None
swa_n = 0
step_size_max = 3*step_size
step_size_min = 0.25*step_size
lol =1
# Optimize
for itt in range(itt_max):

    # generate samples from q
    zs = np.random.normal(0, 1, size=(num_params, num_samples))
    samples, grad_correction = reparametrize(zs, means, log_sigmas, lol)

    # evaluate gradient of log p (does grad_log_prob support vectorization??) and gradient of log q
    try:
        log_p_grad = np.array([fit.grad_log_prob(s) for s in samples.T])
    except Exception:
        print(itt)
        print(params_swa)
        print(params)
        exit()

    entropy_grad = compute_entropy_grad(log_sigmas)

    # compute gradients wrt. mean and log_sigma
    mean_grad = np.mean(log_p_grad, axis=0)
    log_sigma_grad =np.mean(log_p_grad*grad_correction, axis=0) + entropy_grad

    # evaluate ELBO
    #log_p = np.array([fit.log_prob(theta_i) for theta_i in thetas.T])
    #ELBO = np.mean(log_p) + compute_entropy(log_sigmas)

    # take gradient step
    means += step_size*mean_grad
    log_sigmas += step_size*log_sigma_grad
    params = [means, log_sigmas]
    step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
    itt+1, itt_max+1, 8000, 250, params_swa, swa_n)
    #params = params_swa
    means = params[0]
    log_sigmas = params[1]


    #  transform back to constrained space
    sigmas = np.exp(log_sigmas)


# ### Prepare sample from each of the three posterior distributions


# HMC
la = fit.extract(permuted=True)
stan_eta = la['eta']
stan_tau = la['tau']
stan_mu = la['mu']
stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta

# Stan-VB
fit_vb_samples = np.array(fit_vb['sampler_params']).T
stan_vb_mu = fit_vb_samples[:, 0]
stan_vb_tau = fit_vb_samples[:, 1]
stan_vb_eta = fit_vb_samples[:, 2:10]
stan_vb_theta = fit_vb_samples[:, 10:18]


# VB
Q = 5000
vb_mu = np.random.normal(means[0], sigmas[0], size=(Q))
vb_tau = np.exp(np.random.normal(means[1], sigmas[1], size=(Q)))
vb_eta = np.random.normal(means[2:, None], sigmas[2:, None], size=(J, Q)).T
vb_theta = vb_mu[:, None] + vb_tau[:, None]*vb_eta


# ### Plot
snb.set(font_scale=1.5)
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.hist(stan_mu, 50, histtype='step', linewidth=3, label='HMC', density=True);
plt.hist(stan_vb_mu, 50, histtype='step', linewidth=3, label='Stan-VB', density=True);
plt.hist(vb_mu, 50, histtype='step', linewidth=3, label='Python-VB', density=True);
plt.legend()
plt.xlabel('$\mu$')
plt.title('Paramter: $\mu$', fontweight='bold')

plt.subplot(1, 2, 2)
plt.hist(np.log(stan_tau), 50, histtype='step', linewidth=3, label='HMC', density=True);
plt.hist(np.log(stan_vb_tau), 50, histtype='step', linewidth=3, label='Stan-VB', density=True);
plt.hist(np.log(vb_tau), 50, histtype='step', linewidth=3, label='Python-VB', density=True);
plt.legend()
plt.xlabel('$\ln \\tau$')
plt.title('Paramter: $\ln \\tau$', fontweight='bold')

plt.figure(figsize=(20, 30))
for j in range(J):
    plt.subplot(4, 2, 1 + j)
    plt.hist(stan_theta[:, j], 50, histtype='step', linewidth=3, label='HMC', density=True);
    plt.hist(stan_vb_theta[:, j], 50, histtype='step', linewidth=3, label='Stan-VB', density=True);
    plt.hist(vb_theta[:, j], 50, histtype='step', linewidth=3, label='Python-VB', density=True);
    plt.legend()
    plt.xlabel('$\\theta_{%d}$' % (j+1))
    plt.title('Paramter: $\\theta_{%d}$' % (j+1), fontweight='bold')


# ### Funnel
plt.figure(figsize=(10, 6))
plt.plot(np.log(stan_tau), stan_theta[:, 0], 'b.', label='HMC')
plt.plot(np.log(vb_tau), vb_theta[:, 0], 'r.', label='Python-VB')
plt.plot(np.log(stan_vb_tau), stan_vb_theta[:, 0], 'g.', label='Stan-VB')

plt.xlabel('Log $\\tau$')
plt.ylabel('$\\theta_1$')
plt.legend()
plt.show()
