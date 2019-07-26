
import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy
numpy.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=4)
import autograd.numpy as np
import seaborn as snb
import pystan
import scipy
from scipy import stats

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule

from helper import compute_entropy, gaussian_entropy, reparametrize, expectation_iw
from autograd import grad
from arviz import psislw

np.set_printoptions(precision=3)



linear_regression_code= """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    vector[N] y;
}


parameters{
real<lower=0> sigma;
vector[K] w;
}

model{
sigma ~ gamma(0.5, 0.5);
w ~ normal(0, 1);
y ~ normal(X*w, sigma);

}
generated quantities{
real log_joint_density;
log_joint_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0,1) + gamma_lpdf(sigma|0.5,0.5) + log(sigma);
}

"""

N = 200
K= 10
M = 1


mean = np.zeros((K,))
cov = np.ones((K,K))
X= np.random.multivariate_normal(mean, cov, size=N)

#Y = np.dot(X, W)
print(X.shape)
W = np.random.normal(0,1, (K,M))

sigma_0 = np.random.gamma(0.5, 0.5, M)
y_mean= X@W
Y = y_mean + np.random.normal(0, sigma_0[0], (N,M))

for i in range(1,M+1):
    model_data= {'N':N,
       'K':K,
       'y':Y[:,0],
       'X':X}

    sm = pystan.StanModel(model_code=linear_regression_code)
    #fit_hmc = sm.sampling(data=model_data)

    fit_vb = sm.vb(data=model_data, iter =100000, tol_rel_obj=1e-4, output_samples=4000)
# ### Run ADVI in Python
    np.random.seed(123)
    # use analytical gradient of entropy
    compute_entropy_grad = grad(compute_entropy)

    # settings
    step_size = 2e-2
    itt_max = 10000
    num_samples = 1

    num_params = K+1
    means = np.zeros((num_params,1))
    sigmas = np.ones((num_params,1))
    log_sigmas = np.log(sigmas)

    params = [means, sigmas]
    params_swa = None

    means_clr, log_sigmas_clr = means.copy(), sigmas.copy()
    params_constant_lr = [means_clr, log_sigmas_clr]

    lr_t = []
    params_swa = None
    swa_n = 0
    step_size_max = 3*step_size
    step_size_min = 0.25*step_size
    lr_constant = 0.01

    lr_t = []
    tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    eta = [1, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]


    for itt in range(itt_max):
        zs = np.random.normal(0, 1, size=(num_params, num_samples))
        samples, grad_correction = reparametrize(zs, means, log_sigmas)
        samples_clr, grad_correction_clr = reparametrize(zs, means_clr, log_sigmas_clr)

    # evaluate gradient of log p (does grad_log_prob support vectorization??) and gradient of log q
        try:
            log_p_grad = np.array([fit.grad_log_prob(s) for s in samples.T])
            ##log_p_grad_clr = np.array([fit.grad_log_prob(s) for s in samples_clr.T])
        except Exception:
            print(itt)
            print(params_swa)
            print(params)
            print('lol')


        # compute gradients wrt. mean and log_sigma
        mean_grad = np.mean(log_p_grad, axis=0)
        log_sigma_grad =np.mean(log_p_grad*grad_correction, axis=0) + entropy_grad

        mean_grad_clr = np.mean(log_p_grad_clr, axis=0)
        log_sigma_grad_clr =np.mean(log_p_grad_clr*grad_correction_clr, axis=0) + entropy_grad_clr

        # take gradient step
        means += step_size*mean_grad
        log_sigmas += step_size*log_sigma_grad
        params = [means, log_sigmas]
        step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
        itt+1, itt_max+1, 7500, 250, params_swa, swa_n)
        #params = params_swa
        lr_t.append(step_size)
        means = params[0]
        log_sigmas = params[1]
        means_clr += lr_constant*mean_grad_clr
        log_sigmas_clr += lr_constant*log_sigma_grad_clr

        #  transform back to constrained space
        sigmas = np.exp(log_sigmas)
        sigmas_clr = np.exp(log_sigmas_clr)


    # ### Prepare sample from each of the three posterior distributions
    lr_t = np.array(lr_t)

    la = fit.extract(permuted=True)
    stan_w = la['w']
    stan_sigma = la['sigma']
    stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta

    params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
    params_stan_mean = np.mean(params_stan, axis=0)
    params_stan_sq= np.mean(params_stan**2, axis=0)
    # Stan-VB
    fit_vb_samples = np.array(fit_vb['sampler_params']).T
    stan_vb_w = fit_vb_samples[:, 0:100]
    stan_vb_sigma = fit_vb_samples[:, 101]

    # VB-SWA
    means_vb_swa = params_swa[0]
    sigmas_vb_swa = np.exp(params_swa[1])
    means_vb_swa = np.array(means_vb_swa)

    params_vb_means = np.nanmean(fit_vb_samples, axis=0)
    params_vb_std = np.nanstd(fit_vb_samples, axis=0)
    params_vb_sq = np.mean(fit_vb_samples**2, axis=0)

    logq = stats.norm(fit_vb_samples, mean=params_vb_means, scale=params_vb_std)
    logq_sum = np.sum(np.log(logq))
    log_joint_density = la['log_joint_density']
    log_iw = log_joint_density - logq_sum
    psis_lw, K_hat = psislw(log_iw)
    bias_mean = params_vb_means - params_stan_mean
    bias_sq = params_vb_sq - params_stan_sq

    bias_mean_psis = expectation_iw()


    # ### Plot
    snb.set(font_scale=1.1)
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.hist(stan_mu, 50, histtype='step', linewidth=3, label='HMC', density=True);
    plt.hist(stan_vb_mu, 50, histtype='step', linewidth=3, label='Stan-VB', density=True);
    plt.hist(vb_mu_clr, 50, histtype='step', linewidth=3, label='Python-VB', density=True);
    plt.hist(vb_mu_swa, 50, histtype='step', linewidth=3, label='VB-SWA', density=True);
    plt.legend()
    plt.xlabel('$\mu$')
    plt.title('Paramter: $\mu$', fontweight='bold')
