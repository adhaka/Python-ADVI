
import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from GPy.util import choleskies

import sys
sys.path.append('..')

import numpy
numpy.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=4)
import autograd.numpy as np
import seaborn as snb
import pystan
import scipy
from scipy import stats
import pickle
#import scipy.stats as stats

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule, stepsize_linear_weight_averaging, stepsize_linear_adaptive_is_mixing_schedule, rms_prop_gradient, step_size_rms_prop_schedule
from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm, compute_l2_norm_means, compute_l2_norm_Sigmas, \
    compute_R_hat_fn, compute_posterior_moments
from autograd import grad
from arviz import psislw

#from swa_schedules import scale_factor_warm_up, elbo_grad_gaussian
#from swa_schedules import elbo_logit, elbo_grad_gaussian
from helper import compute_posterior_moments
from data_generator import data_generator_linear, data_generator_logistic

from functions import add_noise, scale_factor_warm_up, elbo_grad_logit, elbo_logit, elbo_full_logit, scale_factor_warmup_logit

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-a', type=int, default=1)
parser.add_argument('--N', '-n', type=int, default=2000)
parser.add_argument('--distribution', '-d', type=int, default=1)
parser.add_argument('--posterior', '-p', type=str, default='independent')

args = parser.parse_args()
d = args.distribution
p = args.posterior

algo = 'meanfield'
algo_name='mf'


if args.algorithm ==1:
    algo = 'meanfield'
    algo_name = 'mf'
elif args.algorithm ==2:
    algo = 'fullrank'
    algo_name = 'fr'

# code for approximating family.
if d == 1:
    dist = 'gaussian'
elif d ==2:
    dist = 'student-t'

N = args.N
##  code for linear model with fixed variances .
logistic_reg_code= """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    int<lower=0, upper=1> y[N];
}

parameters{
vector[K] w;
}

model{
y ~ bernoulli_logit(X*w);
}

generated quantities{
real log_joint_density;
log_joint_density = bernoulli_logit_lpmf(y|X*w) + normal_lpdf(w| 0, 1);
}

"""


# Some helper functions
def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(L, Sigma=None):
    return np.sum(np.log(np.diag(L))) + L.shape[0]*(1 + np.log(2*np.pi))

def reparametrize(zs, means, L):
    tmp = np.dot(L, zs)
    #print(means[:,None])
    samples = means[:, None] + tmp
    Linv = np.linalg.inv(L)
    grad_correction = zs.T
    return samples, grad_correction

try:
    a1 = np.load('mean_0_chains_linear.npy')
    print(a1)
    exit()
except:
    pass



SEED= 2019
N = 3000
N_test = 50
K_list = [10]
num_K = len(K_list)
M = 1
w0= 0
N_sim = 4

debug_mode = True
compute_L2_norm = True

K_hat_stan_advi_list = np.zeros((num_K, N_sim))
l2_norm_rms_list = []

itt_max_list = [5000, 10000]

num_itt_list = len(itt_max_list)
#itt_max_list = [5000]
compute_R_hat = True
compute_K_hat = False
step_size_set = False
step_size_rms_set = False

K_hat_vals_rms = np.zeros((num_itt_list, N_sim, num_K))
K_hat_vals_clr = np.zeros((num_itt_list, N_sim, num_K))
K_hat_vals_swa = np.zeros((num_itt_list, N_sim, num_K))
R_hat_list_mean_max = []
R_hat_list_sigma_max = []
R_hat_list_mean_median = []
R_hat_list_sigma_median = []
itt_list = []


for i, itt_max in enumerate(itt_max_list):
    mean_list_i = np.zeros((N_sim, itt_max))
    sigmas_list_i = np.zeros((N_sim, itt_max))
    warmup= int(itt_max/2)
    if warmup % 2 ==1:
        warmup = warmup + 1
    for j in range(num_K):
        np.random.seed(SEED)
        K= K_list[j]
        S= K*(K+1)//2
        vi_params_collapsed = np.zeros((N_sim, itt_max, 2*K))
        vi_params_collapsed2 = np.zeros((N_sim, itt_max, K+S))
        means_all = np.zeros((K * N_sim))
        means_init = np.array([-5. ,0., 11., 15.])
        means_all = np.repeat(means_init, K)
        betas_all = np.ones((K * (K + 1) * N_sim// 2,1 ))
        N_test = 50
        M = 1
        noise_sigma = 1.0
        noise_var = noise_sigma**2

        regression_data= data_generator_logistic(N, K, noise_sigma=noise_sigma, mode=p, seed=SEED)

        X = regression_data['X']
        Y = regression_data['Y']
        W = regression_data['W']
        N_train = N - N_test

        # introduce correlations in x here
        # x_full = (x_full - np.random.normal(0, cov_sd, N_train + N_test)[:, None]) / np.sqrt(1 ** 2 + cov_sd ** 2)
        X = X[:N_train, :]
        X_test = X[N_train:, :]
        Y = Y[:N_train, :]
        Y_test = X[N_train:, :]
        w_mean_vi_list = [0, 10, 15, -5]
        w_mean = 5
        w_sigma = 1.0
        W_mean = np.repeat(w_mean, K)
        W_sigma = np.repeat(w_sigma, K)
        W_cov= W_sigma**2

        model_data = {'N': N_train,
                      'K': K,
                      'y': Y[:, 0],
                      'X': X
                      }

        try:
            sm = pickle.load(open('pickled_results/logistic_reg_chains100.pkl', 'rb'))
        except:
            sm = pystan.StanModel(model_code=logistic_reg_code)
            with open('pickled_results/logistic_reg_chains100.pkl', 'wb') as f:
                pickle.dump(sm, f)

        for n in range(N_sim):
            w_mean_n = w_mean_vi_list[n]

            num_proposal_samples = 10000
            try:
                fit_hmc
            except NameError:
                fit_hmc = sm.sampling(data=model_data, iter=800)

            try:
                fit_vb
            except NameError:
                fit_vb = sm.vb(data=model_data, iter=5000, tol_rel_obj=1e-5, output_samples=num_proposal_samples)
            # ### Run ADVI in Python
            # use analytical gradient of entropy
            compute_entropy_grad = grad(compute_entropy)

            # settings
            #step_size= 1e-5/N
            #itt_max = 30000
            num_samples = 1
            num_samples_swa = 1

            num_params = K
            means1 =  np.ones((num_params,))
            means2 = np.zeros((num_params,))

            means = means_all[n*num_params:(n+1)*num_params]
            betas = betas_all[num_params*(num_params+1)*n//2:num_params*(num_params+1)*(n+1)//2]

            tmpL = np.tril(np.ones((K, K)))
            L = choleskies._flat_to_triang_pure(betas)[0,:]
            Sigma = L @ L.T

            #means = means_list[n]
            #sigmas =  sigmas_list[n]
            params = [means.copy(), betas.copy()]

            means_vb_clr, betas_vb_clr = means.copy(), betas.copy()
            means_vb_swa, betas_vb_swa = means.copy(), betas.copy()
            means_vb_rms, betas_vb_rms = means.copy(), betas.copy()
            L_vb_clr = L.copy()
            L_vb_swa = L.copy()
            L_vb_rms = L.copy()

            params_constant_lr = [means_vb_clr, betas_vb_clr]
            params_rms = [means_vb_rms, betas_vb_rms]

            old_mean_grad = np.zeros_like(means)
            old_beta_grad = np.zeros_like(betas)
            old_L_grad = np.zeros_like(L)

            params_constant_lr = [means_vb_clr, betas_vb_clr]
            params_rms_prop = [means_vb_rms, betas_vb_rms]
            old_mean_grad = np.zeros_like(means)
            old_beta_grad = np.zeros_like(betas)
            old_L_grad = np.zeros_like(L)
            params_swa = None
            swa_n = 0
            start_swa = False
            step_size_rms = 0.1
            params_swa_list =[]
            params_clr_list = []
            lr_t = np.zeros((itt_max,1))
            tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
            eta = [5, 1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 5e-6, 1e-6, 1e-7, 1e-8]

            l2_norm_swa = []
            l2_norm_clr = []
            l2_norm_rms = []
            l2_norm_clr2 = []

            l2_norm_hmc_swa = []
            l2_norm_hmc_clr = []
            l2_norm_hmc_rms = []
            l2_norm_hmc_clr2 = []

            l2_norm_means_swa = []
            l2_norm_means_clr = []
            l2_norm_means_rms = []
            l2_norm_means_clr2 = []

            l2_norm_means_hmc_swa = []
            l2_norm_means_hmc_clr = []
            l2_norm_means_hmc_rms = []
            l2_norm_means_hmc_clr2 = []

            l2_norm_sigmas_swa = []
            l2_norm_sigmas_clr = []
            l2_norm_sigmas_rms = []
            l2_norm_sigmas_clr2 = []

            l2_norm_sigmas_hmc_swa = []
            l2_norm_sigmas_hmc_clr = []
            l2_norm_sigmas_hmc_rms = []
            l2_norm_sigmas_hmc_clr2 = []

            s_mean= 0
            s_log_var=0
            # number of iterations when to begin sw averaging.
            start_swa_iter= 2000
            elbo_threshold_swa = 0.08
            elbo_prev= 100
            elbo_diff_list = []
            elbo_list = []
            elbo_mean_list = []
            s_prev=None
            if step_size_set is False:
                scale_factor= scale_factor_warmup_logit(X, Y, mode='clr', warmup_iters=200)
                print(scale_factor)
                #step_size_rms = scale_factor_warm_up_logit(X, Y, mode='rms', warmup_iters=200)
                step_size_rms = 0.1
                #scale_factor = scale_factor/20000
                step_size_set = True

            if step_size_rms_set is False:
                step_size_rms = scale_factor_warmup_logit(X, Y, mode='rms', warmup_iters=200)
                step_size_rms_set = True

            step_size = scale_factor
            #step_size_rms = scale_factor/10
            #step_size = 0.09
            #step_size = scale_factor
            step_size_max = 1.005*step_size
            step_size_min = 0.995*step_size

            for itt in range(itt_max):
                zs = np.random.normal(0, 1, size=(num_params, num_samples))
                #zs_swa = np.random.normal(0, 1, size=(num_params, num_samples_swa))

                dof = 4
                zs_t = np.random.standard_t(dof, size=(num_params, num_samples))
                zs_t_swa = np.random.standard_t(dof, size=(num_params, num_samples_swa))
                samples, grad_correction = reparametrize(zs, means, L)
                samples_rms, grad_correction_rms = reparametrize(zs, means_vb_rms, L_vb_rms)
                samples_clr, grad_correction_clr = reparametrize(zs, means_vb_clr, L_vb_clr)
                #samples_swa2, grad_correction_swa2 = reparametrize(zs, means_vb_clr2, log_sigmas_vb_clr2)
                #samples, grad_correction = reparametrize(zs_swa, means, log_sigmas)

                #log_p_grad1 = np.array([fit_hmc.grad_log_prob(s)])
                # evaluate gradient of log p (does grad_log_prob support vectorization??) and gradient of log q
                log_p_grad = np.array([fit_hmc.grad_log_prob(s) for s in samples.T])
                log_p_grad_rms = np.array([fit_hmc.grad_log_prob(s) for s in samples_rms.T])
                log_p_grad_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])

                log_p = np.array([fit_hmc.log_prob(s) for s in samples.T])
                elbo= np.mean(log_p) + compute_entropy(L)

                log_p_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])
                elbo_clr = np.mean(log_p_clr) + compute_entropy(L_vb_clr)

                log_p_rms = np.array([fit_hmc.grad_log_prob(s) for s in samples_rms.T])
                elbo_rms = np.mean(log_p_rms) + compute_entropy(L_vb_rms)
                entropy_grad_swa = np.diag(1./np.diag(L_vb_swa))
                entropy_grad_clr = np.diag(1./np.diag(L_vb_clr))
                entropy_grad_rms = np.diag(1./np.diag(L_vb_rms))
                #print(log_p_grad)
                #print( np.mean(log_p_grad*grad_correction, axis=0) + entropy_grad)
                #print(log_p_grad_analytical )
                # compute gradients wrt. mean and log_sigma
                mean_grad = np.mean(log_p_grad, axis=0)
                mean_grad_clr = np.mean(log_p_grad_clr, axis=0)
                mean_grad_rms = np.mean(log_p_grad_rms, axis=0)

                L_grad_clr = (log_p_grad_clr.T @ grad_correction_clr) * tmpL + entropy_grad_clr
                L_grad = (log_p_grad.T @ grad_correction) * tmpL + entropy_grad_swa
                L_grad_rms = (log_p_grad_rms.T @ grad_correction_rms) * tmpL + entropy_grad_rms

                mean_grads_running_dot_product = np.mean(mean_grad * old_mean_grad, axis=0)
                sigma_grads_running_dot_product = np.mean(L_grad * old_L_grad)

                old_mean_grad = mean_grad
                old_L_grad = L_grad
                # s_mean += mean_grads_running_dot_product
                # s_log_var += sigma_grads_running_dot_product
                criterion1 = mean_grads_running_dot_product < 0
                criterion2 = sigma_grads_running_dot_product < 0
                criterion3 = np.abs(elbo_prev - elbo) < np.abs(elbo_threshold_swa * elbo_prev)
                #print('old mean gradient')
                #print(old_mean_grad)
                mean_grads_running_dot_product = np.mean(mean_grad*old_mean_grad)
                sigma_grads_running_dot_product = np.mean(old_L_grad*L_grad)

                print(step_size)
                means_vb_clr += step_size * mean_grad_clr
                L_vb_clr += 0.5 * step_size * L_grad_clr
                betas_vb_clr = choleskies._triang_to_flat_pure(L_vb_clr[None, :])
                betas = choleskies._triang_to_flat_pure(L[None, :])
                # betas_vb_swa = np.reshape(L_vb_swa@L_vb_swa.T, (-1,1))
                params = [means, betas]
                params_rms_prop = [means_vb_rms, L_vb_rms.flatten()]
                # step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
                # itt+1, itt_max+1, start_swa_iter, 80, params_swa, swa_n)
                swa_weight = 1.
                step_size, params_swa, swa_n = stepsize_linear_weight_averaging(params, step_size, step_size_min,
                                                                                step_size_max,
                                                                                itt + 1, itt_max + 1, start_swa_iter,
                                                                                200, params_swa, swa_n, weight=1.1,
                                                                                pmz='std')
                rho, s = rms_prop_gradient(itt + 1, mean_grad_rms, L_grad_rms.flatten(), s_prev, step_size_rms)
                s_prev = s

                means_vb_rms = means_vb_rms + rho[:K] * mean_grad_rms
                variance_grads = rho[K:].copy()
                L_vb_rms = L_vb_rms + np.reshape(variance_grads, L_grad_rms.shape) * L_grad_rms
                # step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
                # itt+1, itt_max+1, 2, 1, params_swa, swa_n)

                old_means = means
                old_L = L

                means += step_size * mean_grad
                L +=  step_size * L_grad
                betas = choleskies._triang_to_flat_pure(L[None, :])
                params = [means, betas]

                means_vb_swa = params_swa[0].copy()
                means_vb_swa = np.asarray(means_vb_swa)
                L_vb_swa = choleskies._flat_to_triang_pure(np.asarray(params_swa[1]))[0,:]
                betas_vb_swa = np.asarray(params_swa[1])

                if itt ==300:
                    print(means)
                    print(means_vb_swa)
                    print(L)
                    print(L_vb_swa)
                    print(params_swa)
                    print(params)
                    #print(betas)
                    #exit()

                la = fit_hmc.extract(permuted=True)
                hmc_w = la['w']
                # stan_sigma = la['sigma']
                # stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta
                params_hmc = hmc_w
                # params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
                params_hmc_mean = np.mean(params_hmc, axis=0)
                params_hmc_sq = np.mean(params_hmc ** 2, axis=0)
                params_hmc_sigmas = np.std(params_hmc, axis=0)

                if compute_L2_norm:
                    l2_norm_swa_i = compute_l2_norm(W_mean, W_sigma, means_vb_swa, L_vb_swa)
                    l2_norm_rms_i = compute_l2_norm(W_mean, W_sigma, means_vb_rms, L_vb_rms)
                    l2_norm_clr2_i = compute_l2_norm(W_mean, W_sigma, means_vb_clr, L_vb_clr)
                    mean_i = means_vb_clr[0]
                    mean_list_i[n, itt] = mean_i
                    sigma_i = L_vb_clr[0,0]
                    sigmas_list_i[n, itt] = sigma_i
                    #print(mean_i)

                    l2_norm_swa_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_swa, L_vb_swa)
                    l2_norm_rms_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_rms, L_vb_rms)
                    l2_norm_clr_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_clr, L_vb_clr)

                    l2_norm_hmc_swa_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_swa)
                    l2_norm_hmc_swa_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, np.diag(L_vb_swa))
                    l2_norm_hmc_rms_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_rms)
                    l2_norm_hmc_rms_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, np.diag(L_vb_rms))
                    l2_norm_hmc_clr_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_clr)
                    l2_norm_hmc_clr_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, np.diag(L_vb_clr))

                    l2_norm_means_hmc_swa.append(l2_norm_hmc_swa_means_i)
                    l2_norm_means_hmc_rms.append(l2_norm_hmc_rms_means_i)
                    l2_norm_means_hmc_clr.append(l2_norm_hmc_clr_means_i)

                    l2_norm_sigmas_hmc_swa.append(l2_norm_hmc_swa_sigmas_i)
                    l2_norm_sigmas_hmc_rms.append(l2_norm_hmc_rms_sigmas_i)
                    l2_norm_sigmas_hmc_clr.append(l2_norm_hmc_clr_sigmas_i)

                    l2_norm_hmc_swa.append(l2_norm_swa_hmc_i)
                    l2_norm_hmc_rms.append(l2_norm_rms_hmc_i)
                    l2_norm_hmc_clr.append(l2_norm_clr_hmc_i)

                elbo_prev= elbo
                elbo_list.append(elbo)
                elbo_mean_list.append(np.mean(elbo_list[-20:-1]))

                if itt > warmup:
                    means_vb_clr = add_noise(means_vb_clr, sd=0.001)

                if compute_R_hat:
                    a2 = np.concatenate((means_vb_clr.flatten(), betas_vb_clr.flatten()), axis=None)
                    vi_params_collapsed2[n, itt] = a2

                # ### Prepare sample from each of the three posterior distributions
                lr_t = np.array(lr_t)


            if compute_K_hat:
                samples_vb_clr2 = np.random.multivariate_normal(means_vb_clr, L_vb_clr@L_vb_clr.T,
                                                                size=num_proposal_samples)
                samples_vb_swa = np.random.multivariate_normal(means_vb_swa, L_vb_swa@L_vb_swa.T,
                                                               size=num_proposal_samples)
                samples_vb_rms = np.random.multivariate_normal(means_vb_rms, L_vb_rms@L_vb_rms.T,
                                                               size=num_proposal_samples)

                q_swa = stats.norm.logpdf(samples_vb_swa, means_vb_swa, L_vb_swa)
                logp_swa = np.array([fit_hmc.log_prob(s) for s in samples_vb_swa])
                #logp_swa = np.array([fit_hmc.])
                log_iw_swa = logp_swa - np.sum(q_swa, axis=1)
                psis_lw_swa, K_hat_swa = psislw(log_iw_swa.T)
                print('K hat statistic for SWA')
                print(K_hat_swa)

                # VB-CLR
                q_clr = stats.norm.logpdf(samples_vb_clr2, means_vb_clr, L_vb_clr)
                logp_clr = np.array([fit_hmc.log_prob(s) for s in samples_vb_clr2])
                log_iw_clr = logp_clr - np.sum(q_clr, axis=1)
                psis_lw_clr, K_hat_clr = psislw(log_iw_clr.T)
                print('K hat statistic for CLR')
                print(K_hat_clr)

                q_rms = stats.norm.logpdf(samples_vb_rms, means_vb_rms, L_vb_rms)
                logp_rms = np.array([fit_hmc.log_prob(s) for s in samples_vb_rms])
                log_iw_rms = logp_rms - np.sum(np.log(q_rms), axis=1)
                psis_lw_rms, K_hat_rms = psislw(log_iw_rms.T)
                print('K hat statistic for RMS-Prop')
                print(K_hat_rms)
                K_hat_vals_clr[i, n, j] = K_hat_clr
                K_hat_vals_rms[i, n, j] = K_hat_rms
                K_hat_vals_swa[i, n, j] = K_hat_swa

                #  VB-SWA
                #means_vb_swa = params_swa[0]
                #sigmas_vb_swa = np.exp(params_swa[1])
                #means_vb_swa = np.array(means_vb_swa)
                #samples_vb_swa = np.random.multivariate_normal(means_vb_swa, np.diag(sigmas_vb_swa), size=num_proposal_samples)
                #samples_vb_rms = np.random.multivariate_normal(means_vb_rms, np.diag(sigmas_vb_rms), size=num_proposal_samples)
                #q_swa = stats.norm.pdf(samples_vb_swa, means_vb_swa, sigmas_vb_swa)
                #logp_swa = np.array([fit_hmc.log_prob(s) for s in samples_vb_swa])
                #log_iw_swa = logp_swa - np.sum(np.log(q_swa), axis=1)
                #psis_lw_swa, K_hat_swa = psislw(log_iw_swa.T)
                #print('K hat statistic for SWA')
                #print(K_hat_swa)
                #K_hat_swa_list[j,n] = K_hat_swa
                # VB-CLR
                #
            print('hii')
            #plt.figure(figsize=(20,6))

    chains_clr2 = vi_params_collapsed2
    #chains2 = vi_params_collapsed[:, warmup:, :]
    #chains_clr2_noisy = add_noise_chains(chains_clr2, sd=0.1)
    _, R_hat2 = compute_R_hat_fn(chains_clr2[:, :, :], warmup)
    R_hat_list_mean_max.append(np.max(R_hat2[:K]))
    R_hat_list_sigma_max.append(np.max(R_hat2[K:]))
    R_hat_list_mean_median.append(np.median(R_hat2[:K]))
    R_hat_list_sigma_median.append(np.median(R_hat2[K:]))


fit_vb_samples = np.array(fit_vb['sampler_params']).T
stan_vb_w = fit_vb_samples[:, 0:K]
params_vb_stan_means = np.mean(stan_vb_w, axis=0)
params_vb_stan_std = np.std(stan_vb_w, axis=0)
params_vb_stan_sq = np.mean(stan_vb_w ** 2, axis=0)

l2_norm_vb_stan_i = compute_l2_norm(params_vb_stan_means, params_vb_stan_std, params_hmc_mean, np.diag(params_hmc_sigmas))
l2_norm_vb = np.repeat(l2_norm_vb_stan_i, itt_max)
logq = stats.norm.pdf(stan_vb_w, params_vb_stan_means, params_vb_stan_std)
logq_sum = np.sum(np.log(logq), axis=1)
# log_joint_density = la['log_joint_density']
stan_vb_log_joint_density = fit_vb_samples[:, (K + 1)]
log_iw = stan_vb_log_joint_density - logq_sum
#print(np.max(log_iw))

print(log_iw.shape)
psis_lw, K_hat_stan = psislw(log_iw.T)
print(psis_lw.shape)
print('K hat statistic for Stan ADVI:')
print(K_hat_stan)

la = fit_hmc.extract(permuted=True)
hmc_w = la['w']
#stan_sigma = la['sigma']
# stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta

params_hmc = hmc_w
#params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
params_hmc_mean = np.mean(params_hmc, axis=0)
params_hmc_sq= np.mean(params_hmc**2, axis=0)
params_hmc_sigmas= np.std(params_hmc, axis=0)

l2_norm_vb_stan_means_i = compute_l2_norm_means(params_vb_stan_means, W_mean)
l2_norm_vb_stan_sigmas_i = compute_l2_norm_Sigmas(params_vb_stan_std, W_cov)
l2_norm_vb_stan_means = np.repeat(l2_norm_vb_stan_means_i, itt_max)
l2_norm_vb_stan_sigmas = np.repeat(l2_norm_vb_stan_sigmas_i, itt_max)

chains = vi_params_collapsed
chains_clr2 = vi_params_collapsed2
chains2 = vi_params_collapsed[:,warmup:,:]
R_hat2 = compute_R_hat_fn(chains_clr2, warmup)
print(R_hat2)

if debug_mode:
    print(W_mean)
    print(np.diag(W_sigma))
    print(params_hmc_mean)
    print(params_hmc_sigmas)
    print(params_vb_stan_means)
    print(params_vb_stan_std)
    print(means_vb_clr)
    print(L_vb_clr)
    print(means_vb_rms)
    print(L_vb_rms)
    print(means_vb_swa)
    print(L_vb_swa)
    print(vi_params_collapsed.shape)


plt.figure()
plt.plot(mean_list_i[0,:], 'r-', label='Chain 1')
plt.plot(mean_list_i[1,:], 'g-', label='Chain 2')
plt.plot(mean_list_i[2,:], 'm-', label='Chain 3')
plt.plot(mean_list_i[3,:], 'b-', label='Chain 4')
plt.plot(np.repeat(params_hmc_mean[0], len(mean_list_i[1,:])), 'k-', label='HMC posterior')
plt.ylim((3,6))
plt.xlabel('Iterations')
plt.ylabel('Means')
plt.legend()

plt.savefig('plots/Log_Reg_vi_chains_means_FR_10D.pdf')
np.save('mean_0_chains', mean_list_i )
print(mean_list_i)

plt.figure()

plt.plot(sigmas_list_i[0,:], 'r-', label='Chain 1')
plt.plot(sigmas_list_i[1,:], 'g-', label='Chain 2')
plt.plot(sigmas_list_i[2,:], 'm-', label='Chain 3')
plt.plot(sigmas_list_i[3,:], 'b-', label='Chain 4')
plt.xlabel('Iterations')
plt.ylabel('Variances')
plt.plot(np.repeat(params_hmc_sigmas[0], len(sigmas_list_i[1,:])), 'k-', label='HMC posterior')
plt.ylim((-0.2, 1.2))

plt.legend()
plt.savefig('plots/Log_Reg_vi_chains_sigma_0_FR_10D.pdf')

plt.figure()
plt.plot(itt_max_list, R_hat_list_mean_max,  'ro')
plt.plot(itt_max_list, R_hat_list_sigma_max,  'bo')
plt.plot(itt_max_list, R_hat_list_mean_max,  'r-', label='R_hat_mean')
plt.plot(itt_max_list, R_hat_list_sigma_max,  'b-', label='R_hat_sigma')
plt.plot(itt_max_list, R_hat_list_mean_median,  'r+')
plt.plot(itt_max_list, R_hat_list_sigma_median,  'b+')
plt.plot(itt_max_list, R_hat_list_mean_median,   'r-', label='R_hat_mean', alpha=0.5)
plt.plot(itt_max_list, R_hat_list_sigma_median, 'b-', label='R_hat_sigma', alpha=0.5)

plt.legend()
plt.savefig('plots/Log_Reg_R_hat_vs_iters_FR_10D.pdf')

plt.figure()
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_clr, axis=2), axis=1),  'go')
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_rms, axis=2), axis=1),  'ro')
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_swa, axis=2), axis=1),  'bo')
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_clr, axis=2), axis=1),  'g-', label='CLR')
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_rms, axis=2), axis=1),  'r-', label='RMS-prop')
plt.plot(itt_max_list, np.median(np.mean(K_hat_vals_swa, axis=2), axis=1),  'b-', label='SWA')

plt.legend()
plt.savefig('plots/Log_Reg_K_hat_vs_iters_FR_10D.pdf')
plt.figure()


plt.figure()
plt.plot(l2_norm_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_hmc_clr, 'g-', label='CLR')
plt.plot(l2_norm_hmc_swa, 'b-', label='SWA')
plt.yscale('log')
plt.legend()
plt.savefig('Log_Reg_FR_10D_2000N_hmc_Gelbo.pdf')



plt.figure()
plt.plot(l2_norm_means_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_means_hmc_clr, 'g-', label='CLR')
plt.plot(l2_norm_means_hmc_swa, 'b-', label='SWA')

#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Log_Reg_FR_mean_10D_2000N_hmc_Gelbo.pdf')
#plt.show()

plt.figure()
plt.plot(l2_norm_sigmas_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_sigmas_hmc_clr, 'g-', label='CLR')
plt.plot(l2_norm_sigmas_hmc_swa, 'b-', label='SWA')
#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Log_Reg_FR_sigmas_10D_2000N_hmc_Gelbo.pdf')
