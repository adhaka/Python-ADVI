
'''
file to plot K hat values vs D where D is the dimensionality of the variational parameters for a
linear gaussian model where the covariates are correlated, and mean field approximation is
insufficient and hence we use full rank approximation here.

'''


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
import pickle
import GPy

import argparse
from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule

from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm
from autograd import grad
from arviz import psislw

np.set_printoptions(precision=3)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-a', type=int, default=1)
parser.add_argument('--N', '-n', type=int, default=1000)
parser.add_argument('--iters', '-i', type=int, default=80000)
parser.add_argument('--samplesgrad', '-s', type=int, default=1)
parser.add_argument('--sampleselbo', '-e', type=int, default=1)
parser.add_argument('--evalelbo', '-l', type=int, default=100)

args = parser.parse_args()

max_iters = args.iters
algo = 'meanfield'
algo_name='mf'
gradsamples = args.samplesgrad
elbosamples = args.sampleselbo
evalelbo = args.evalelbo

if args.algorithm ==1:
    algo = 'meanfield'
    algo_name = 'mf'
elif args.algorithm ==2:
    algo = 'fullrank'
    algo_name = 'fr'

N_user= args.N
## code for general linear model without any constraints and gamma prior for std dev.
##  code for linear model with fixed variances .
logistic_reg_correlated_variance_code= """
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
#log_joint_density = bernoulli_logit_lpmf(y|X*w) + normal_lpdf(w| 0, 1);
}
"""

np.random.seed(209)

logit = lambda x: 1./ (1 +np.exp(-x))

# Some helper functions
def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(log_sigma):
    return np.sum(gaussian_entropy(log_sigma))

def reparametrize(zs, means, log_sigmas):
    samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
    log_sigma_grad = (np.exp(log_sigmas[:, None])*zs).T
    return samples, log_sigma_grad


def elbo_data_term(x, zs, means, log_sigmas):
    samples, log_sigma_grad= reparametrize(zs, means, log_sigmas)
    y = samples.T @ x
    p = logit(y)
    elbo_per_point = np.mean(np.log(p)*y + np.log(1.-p)*(1. -y), axis=1)
    elbo_full = np.sum(elbo_per_point, axis=0)
    return elbo_full + gaussian_entropy(log_sigmas)


logit = lambda x: 1./ (1 +np.exp(-x))

N_train = N_user
N = N_user
K_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
K_list = [5, 10, 20, 30, 40, 50, 60]
num_K = len(K_list)
N_sim= 7

K_hat_stan_advi_list = np.zeros((num_K, N_sim))
debug_mode = True


for j in range(num_K):
    for n in range(N_sim):
        N_train = N_user
        N_test = 50
        K = K_list[j]
        M = 1
        w0 = 0

        #mean = np.zeros((K,))
        #cov = np.ones((K, K))

        cov_vector = np.array([0, 0.2, 0.5, 1., 1.5, 2, 2.5, 3., 5.])
        J = cov_vector.size
        cov_sd = 2.2

        x_full = np.zeros((N_train + N_test, K))

        for k in np.arange(K):
            x_full[:, k] = np.random.normal(0, 1, N_train + N_test)

        # introduce correlations in x here
        #x_full = (x_full - np.random.normal(0, cov_sd, N_train + N_test)[:, None]) / np.sqrt(1 ** 2 + cov_sd ** 2)
        X = x_full[:N_train, :]
        X_test = x_full[N_train:, :]

        Z= np.linspace(-1,1, K)
        Z = Z[:,np.newaxis]
        rbf_kernel = GPy.kern.RBF(lengthscale=1, input_dim=1)
        covar= rbf_kernel.K(Z)
        covar2 = np.eye(K) + (np.ones((K,K))*0.7 - np.eye(K)*0.7)

        w_mean = 0
        w_mean = 4
        w_sigma= 0.5
        #w_sigma = 1
        w_mean_true = w_mean
        w_sigma_true = w_sigma
        W_mean = np.ones((K,))*4
        W_cov = covar2*w_sigma
        W = np.random.multivariate_normal(W_mean, W_cov, 1).T
        y_mean = x_full @ W
        p_full = logit(y_mean)
        y_full = np.random.binomial(n=1, p=p_full)
        Y = y_full[:N_train]
        Y_test = y_full[N_train:]

        model_data= {'N':N_train,
       'K':K,
       'y':Y[:,0],
       'X':X
       }

        #sm = pystan.StanModel(model_code=logistic_reg_correlated_variance_code)
        try:
            sm = pickle.load(open('model_logistic_correlated_11.pkl', 'rb'))
        except:
            sm = pystan.StanModel(model_code=logistic_reg_correlated_variance_code)
            with open('model_logistic_correlated_11.pkl', 'wb') as f:
                pickle.dump(sm, f)

        num_proposal_samples = 6000
        #fit_hmc = sm.sampling(data=model_data, iter=600)
        fit_vb = sm.vb(data=model_data, iter=max_iters, tol_rel_obj=1e-4, output_samples=num_proposal_samples,
                       algorithm=algo, grad_samples =gradsamples, elbo_samples=elbosamples, eval_elbo=evalelbo)

        # ### Run ADVI in Python
        # use analytical gradient of entropy
        compute_entropy_grad = grad(compute_entropy)
        # ### Run ADVI in Python
        # settings
        step_size = 4e-7
        step_size= 1e-5/N_train
        itt_max = 2000
        num_samples = 1
        num_params = K
        means = np.ones((num_params,))
        sigmas = np.ones((num_params,))
        log_sigmas = np.log(sigmas)
        params = [means, sigmas]
        means_vb_clr, log_sigmas_vb_clr = means.copy(), log_sigmas.copy()
        params_constant_lr = [means_vb_clr, log_sigmas_vb_clr]
        means2, log_sigmas2 = means.copy(), log_sigmas.copy()
        old_mean_grad = np.zeros_like(means)
        old_log_sigma_grad = np.zeros_like(log_sigmas)

        lr_t = []
        params_swa = None
        swa_n = 0
        step_size_max = 4*step_size
        step_size_min = 0.25*step_size

        lr_constant = 0.0000004
        step_size= 1e-5/N_train

        params_swa_list =[]
        params_clr_list = []
        lr_t = []
        tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
        eta = [1, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]

        l2_norm_swa = []
        l2_norm_clr = []
        l2_norm_swa2 = []
        K_hat_clr_list = []
        K_hat_swa_list = []
        s_mean= 0
        s_log_var=0
        start_swa_iter= 200
        elbo_threshold_swa = 0.10
        elbo_prev= 100
        elbo_diff_list = []
        elbo_list = []


        for itt in range(itt_max):
            zs = np.random.normal(0, 1, size=(num_params, num_samples))
            samples, grad_correction = reparametrize(zs, means, log_sigmas)
            samples_clr, grad_correction_clr = reparametrize(zs, means_vb_clr, log_sigmas_vb_clr)
            samples2, grad_correction2 = reparametrize(zs, means2, log_sigmas2)

        #s = np.ones((10,))
        #log_p_grad1 = np.array([fit_hmc.grad_log_prob(s)])

        # evaluate gradient of log p (does grad_log_prob support vectorization??) and gradient of log q
        # log_p_grad = np.array([fit_hmc.grad_log_prob(s) for s in samples.T])
        # log_p_grad_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])
        # log_p_grad_swa_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples2.T])
        #
        # log_p = np.array([fit_hmc.grad_log_prob(s) for s in samples.T])
        # elbo= np.mean(log_p) + compute_entropy(log_sigmas)
        #
        # log_p_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])
        # elbo_clr = np.mean(log_p_clr) + compute_entropy(log_sigmas_vb_clr)
        #
        # entropy_grad = compute_entropy_grad(log_sigmas)
        # entropy_grad_clr = compute_entropy_grad(log_sigmas_vb_clr)
        # entropy_grad_swa_clr = compute_entropy_grad(log_sigmas_vb_clr)
        # # compute gradients wrt. mean and log_sigma
        # mean_grad = np.mean(log_p_grad, axis=0)
        # log_sigma_grad =np.mean(log_p_grad*grad_correction, axis=0) + entropy_grad
        #
        # # Inner product of gradients
        # print('mean gradient')
        # print(mean_grad)
        # print('old mean gradient')
        # print(old_mean_grad)
        #
        # mean_grads_running_dot_product = np.mean(mean_grad*old_mean_grad)
        # sigma_grads_running_dot_product = np.mean(log_sigma_grad*old_log_sigma_grad)
        #
        # s_mean += mean_grads_running_dot_product
        # s_log_var += sigma_grads_running_dot_product
        # old_mean_grad = mean_grad
        # old_log_sigma_grad = log_sigma_grad
        #
        # if (itt+1) % 10== 0:
        #     print(mean_grads_running_dot_product)
        #     print(sigma_grads_running_dot_product)
        #
        #
        # criterion1 = mean_grads_running_dot_product < 0
        # criterion2 = sigma_grads_running_dot_product < 0
        # criterion3 = np.abs(elbo_prev - elbo) < elbo_threshold_swa*elbo_prev
        # elbo_diff_list.append(elbo - elbo_prev)
        #
        # elbo_diff_median =  np.median(np.array(elbo_diff_list[-21:-1]))
        # elbo_diff_mean = np.mean(np.array(elbo_diff_list[-21:-1]))
        # elbo_diff_last_20 = elbo_diff_list[-20:-1]
        # #elbo_diff_max = np.max(np.array(elbo_diff_list[-21:-1]))
        # elbo_diff_list_abs = [0 for i in elbo_diff_last_20 if i < 0]
        # val1 = len(elbo_diff_list_abs) - np.count_nonzero(np.asarray(elbo_diff_list_abs))
        # criterion4 = val1 > 5

        # if len(elbo_mean_list) > 6:
        #     criterion5 = np.abs(elbo_mean_list[-1] - elbo_mean_list[-2]) < np.abs(elbo_mean_list[-2] - elbo_mean_list[-5])*0.20
        #
        # if criterion1 and criterion2 and criterion4 and criterion5:
        #     start_swa = True
        #     start_swa_iter = itt+1
        #     print(start_swa_iter)
            #print(elbo_diff_list)
            #exit()



        #mean_grad = mean_grad[:, np.newaxis]
        # mean_grad_clr = np.mean(log_p_grad_clr, axis=0)
        # log_sigma_grad_clr =np.mean(log_p_grad_clr*grad_correction_clr, axis=0) + entropy_grad_clr
        #
        # mean_grad_swa_clr = np.mean(log_p_grad_swa_clr, axis=0)
        # log_sigma_grad_swa_clr =np.mean(log_p_grad_swa_clr*grad_correction2, axis=0) + entropy_grad_swa_clr
        # # take gradient step
        # print(step_size)
        # means_vb_clr += lr_constant*mean_grad_clr
        # log_sigmas_vb_clr += lr_constant*log_sigma_grad_clr
        #
        # print(means)
        # print(log_sigmas)
        # params = [means, log_sigmas]
        # #step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
        # #itt+1, itt_max+1, start_swa_iter, 80, params_swa, swa_n)
        #
        # step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
        # itt+1, itt_max+1, start_swa_iter, 200, params_swa, swa_n)
        #
        # step_size_min = step_size
        # step_size_max = step_size+1e-9
        #
        # #step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
        # #itt+1, itt_max+1, 2, 1, params_swa, swa_n)
        # means2 += lr_constant*mean_grad_swa_clr
        # log_sigmas2 += lr_constant*log_sigma_grad_swa_clr
        # means += step_size*mean_grad
        # log_sigmas += step_size*log_sigma_grad
        # #params = params_swa
        # lr_t.append(step_size)
        # #  transform back to constrained space
        # sigmas = np.exp(log_sigmas)
        # sigmas_vb_clr = np.exp(log_sigmas_vb_clr)
        # params_vb_clr  = [means_vb_clr, sigmas_vb_clr]
        # params_vb_swa2 = [means, sigmas]
        #params_clr_list.append(params_vb_clr)
        #params_swa_list.append(params_vb_swa2)
        # means_vb_swa = params_swa[0].copy()
        # sigmas_vb_swa = np.exp(params_swa[1]).copy()
        # l2_norm_clr_i = compute_l2_norm(W_mean, W_sigma, means_vb_clr, sigmas_vb_clr)
        # l2_norm_swa_i = compute_l2_norm(W_mean, W_sigma, means_vb_swa, sigmas_vb_swa)
        # l2_norm_swa2_i = compute_l2_norm(W_mean, W_sigma, means2, np.exp(log_sigmas2))
        # l2_norm_swa.append(l2_norm_swa_i)
        # l2_norm_clr.append(l2_norm_clr_i)
        # l2_norm_swa2.append(l2_norm_swa2_i)
        # elbo_prev= elbo
        # elbo_list.append(elbo)
        # elbo_mean_list.append(np.mean(elbo_list[-20:-1]))


        # samples_vb_swa = np.random.multivariate_normal(means_vb_swa, np.diag(sigmas_vb_swa), size=num_proposal_samples)
        # samples_vb_clr = np.random.multivariate_normal(means_vb_clr, np.diag(sigmas_vb_clr), size=num_proposal_samples)
        #
        # q_swa = stats.norm.pdf(samples_vb_swa, means_vb_swa, sigmas_vb_swa)
        # logp_swa = np.array([fit_hmc.log_prob(s) for s in samples_vb_swa])
        # log_iw_swa = logp_swa - np.sum(np.log(q_swa), axis=1)
        # psis_lw_swa, K_hat_swa = psislw(log_iw_swa.T)
        # print('K hat statistic for SWA')
        # print(K_hat_swa)
        # K_hat_swa_list.append(K_hat_swa)
        #
        # # VB-CLR
        # q_clr = stats.norm.pdf(samples_vb_clr, means_vb_clr, sigmas_vb_clr)
        # logp_clr = np.array([fit_hmc.log_prob(s) for s in samples_vb_clr])
        # log_iw_clr = logp_clr - np.sum(np.log(q_clr), axis=1)
        # psis_lw_clr, K_hat_clr = psislw(log_iw_clr.T)
        # K_hat_clr_list.append(K_hat_clr)
        #
        # print('K hat statistic for CLR')
        # print(K_hat_clr)


        # ### Prepare sample from each of the three posterior distributions
        lr_t = np.array(lr_t)
        # Stan-VB
        fit_vb_samples = np.array(fit_vb['sampler_params']).T
        #fit_summary = fit_vb.extract()
        print(fit_vb_samples.shape)
        print(fit_vb['sampler_param_names'])
        #print(fit_summary)

        print(fit_vb_samples[10,:])
        #log_density = np.array(fit_vb['lp__'])

        stan_vb_w = fit_vb_samples[:, :K]

        p = logit(X@ stan_vb_w.T)

        print(p[:,1])
        log_density2= np.sum(np.log(p)*Y + np.log(1-p+1e-10)*(1.- Y), axis=0)
        print(log_density2.shape)

        params_vb_means = np.mean(stan_vb_w, axis=0)
        params_vb_std = np.std(stan_vb_w, axis=0)
        params_vb_sq = np.mean(stan_vb_w**2, axis=0)

        logq = stats.norm.pdf(stan_vb_w, params_vb_means, params_vb_std)
        logq_sum = np.sum(np.log(logq), axis=1)
        print(logq_sum.shape)
        stan_vb_log_joint_density = fit_vb_samples[:, K]
        log_iw = log_density2 - logq_sum
        psis_lw, K_hat = psislw(log_iw.T)
        K_hat_stan_advi_list[j, n] = K_hat
        #print(psis_lw.shape)
        print('K hat statistic for Stan ADVI:')
        print(K_hat)
#     print(psis_lw[:100])
#
#     bias_mean = params_vb_means - params_hmc_mean
#     bias_sq = params_vb_sq - params_hmc_sq
#
#     print(bias_mean)
#     print(bias_sq)
#     print(params_hmc_mean)
#     print(params_hmc_sigmas)
#     print(params_vb_means)
#     print(params_vb_std)
#     print(means_vb_swa)
#     print(sigmas_vb_swa)
#     print(means_vb_clr)
#     print(sigmas_vb_clr)
#     print(np.mean(W))
#     print(log_iw.shape)
#     print(params_vb_means.shape)
#
#     bias_mean_psis = expectation_iw(log_iw[:,None].T, stan_vb_w)
#     print(bias_mean_psis)
#
#     bias_mean_psis_swa = expectation_iw(log_iw_swa[:,None].T, samples_vb_swa)
#     print(bias_mean_psis_swa)
#
#     bias_mean_psis_clr = expectation_iw(log_iw_clr[:,None].T, samples_vb_clr)
#     print(bias_mean_psis_clr)
#
#     l2_norm_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, W_mean, W_sigma)
#     l2_norm_hmc = np.repeat(l2_norm_hmc_i, 1100)
#
#     l2_norm_advi_i = compute_l2_norm(means_vb_swa, np.sqrt(params_vb_std), W_mean, W_sigma)
#     l2_norm_advi = np.repeat(l2_norm_advi_i, 1100)


###################### Plotting L2 norm here #################################

#plt.figure()
#plt.plot(stan_vb_w[:,0], stan_vb_w[:,1], 'mo', label='STAN-ADVI')
#plt.savefig('vb_w_samples.pdf')


            # plt.figure(figsize=(20, 6))
            # plt.plot(K_hat_clr_list[10:], label='CLR')
            # plt.plot(K_hat_swa_list[10:], label='SWA')
            # plt.legend()
            # plt.savefig('Bayesian_Linear_Regression_K_hat1_50_2.pdf')
            # plt.show()

plt.figure()
plt.plot(K_list, np.nanmean(K_hat_stan_advi_list, axis=1), 'r-', alpha=1)
plt.plot(K_list, np.nanmin(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.nanmax(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.xlabel('Dimensions')
plt.ylabel('K-hat')
np.save('K_hat_logistic_correlated_'+algo_name + '_' + str(N) + 'N', K_hat_stan_advi_list)
#plt.ylim((0,5))


plt.legend()
plt.savefig('Logistic_Regression_K_hat_vs_D_correlated_' + algo_name +'_' + str(N) + 'N.pdf')
