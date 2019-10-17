'''
file to plot K hat values vs D where D is the dimensionality of the variational parameters for a
linear gaussian model where the covariates are independent, and thus mean field approximation is
sufficient.

'''



import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from GPy.util import choleskies

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
from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm
from autograd import grad
from arviz import psislw

np.set_printoptions(precision=3)

##  code for linear model with fixed variances .
linear_reg_fixed_variance_code= """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    vector[N] y;
    real<lower=0> sigma;
    real<lower=0> w_sigma;
    real w_mean;
    real b;
}

parameters{
vector[K] w;
}

model{
w ~ normal(w_mean, w_sigma);
y ~ normal(X*w + b, sigma);
}

generated quantities{
real log_joint_density;
log_joint_density = normal_lpdf(y|X*w +b, sigma) + normal_lpdf(w| w_mean, w_sigma) + log(sigma);
}
"""

#np.random.seed(123)


# Some helper functions
def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(log_sigma):
    return np.sum(gaussian_entropy(log_sigma))

def reparametrize(zs, means, log_sigmas):
    samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
    log_sigma_grad = (np.exp(log_sigmas[:, None])*zs).T
    return samples, log_sigma_grad



# K=  2
# M = 1
# w0= 0
#
# cov_vector= np.array([0, 0.2, 0.5, 1., 1.5, 2,2.5, 3.,5.])
# J = cov_vector.size
# cov_sd = 2.2
# N_train = 400
# N_test = 50
# x_full=np.zeros((N_train+N_test, K))
#
# for k in np.arange(K):
#     x_full[:,k] = np.random.normal(0,1, N_train+N_test)
#
# #x_full =  (x_full-np.random.normal(0, cov_sd, N_train+N_test)[:,None]) / np.sqrt(1**2 + cov_sd**2)
# X = x_full[:N_train,:]
# X_test = x_full[N_train:,:]
#
#
# W_mean = np.asarray([2, 2])
# #W_cov = np.asarray([[1.0,0, 0],[0., 1.0, 0], [0, 0, 0.5]])
#
# W_cov = np.asarray([[0.7,0],[0., 0.7]])
#
# W= W_mean
# W = np.random.multivariate_normal(W_mean, W_cov)
# #sigma_0 = np.random.gamma(0.5, 0.5, M)
# sigma_0 = 0.25
#
# y_mean= x_full@W
# Y = y_mean + np.random.normal(0, sigma_0, (N_train,M))



N_train = 5000
N = 5000
K_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
K_list = [5, 10, 19, 30, 40, 50, 60, 70, 80, 90, 100]
num_K = len(K_list)
M = 1
w0= 0
N_sim = 3

debug_mode = True

K_hat_clr_list = np.zeros((num_K, N_sim))
K_hat_swa_list = np.zeros((num_K, N_sim))
K_hat_rms_list = np.zeros((num_K, N_sim))
K_hat_stan_advi_list = np.zeros((num_K, N_sim))


for j in range(num_K):
    for n in range(N_sim):
        K = K_list[j]
        mean = np.zeros((K,))
        cov = np.ones((K,K))
        X_tmp = np.random.multivariate_normal(mean, cov, size=N)
        X = np.concatenate((X_tmp, np.ones((N,1))), axis=1)

        #Y = np.dot(X, W)
        print(X.shape)
        w_mean = 0
        w_mean = 2
        w_sigma= 0.5
        #w_sigma = 1
        w_mean_true = w_mean
        w_sigma_true = w_sigma

        W = np.random.normal(w_mean, w_sigma, (K,M))
        W = np.concatenate((W, np.ones((1,M))*w0), axis=0)

        W_mean = np.repeat(w_mean, K)
        W_sigma = np.repeat(w_sigma, K)

        #sigma_0 = np.random.gamma(0.5, 0.5, M)
        sigma_0 = 0.90
        sigma_0 = 0.45
        y_mean= X@W
        Y = y_mean + np.random.normal(0, sigma_0, (N,M))

        for i in range(1,M+1):
            model_data= {'N':N,
               'K':K,
               'y':Y[:,0],
               'X':X[:,:K],
               'sigma':sigma_0,
               'w_mean':w_mean,
               'w_sigma':w_sigma,
               'b':w0
               }

        #    sm = pystan.StanModel(model_code=linear_reg_fixed_variance_code)
            try:
                sm = pickle.load(open('linear_model.pkl', 'rb'))
            except:
                sm = pystan.StanModel(model_code=linear_reg_fixed_variance_code)
            with open('linear_model.pkl', 'wb') as f:
                pickle.dump(sm, f)

            num_proposal_samples = 7000
            #fit_hmc = sm.sampling(data=model_data, iter=400)
            fit_vb = sm.vb(data=model_data, iter=20000, tol_rel_obj=1e-5, output_samples=num_proposal_samples)
            # ### Run ADVI in Python
            # use analytical gradient of entropy
            compute_entropy_grad = grad(compute_entropy)

            # settings
            step_size = 4e-7
            step_size= 1e-5/N
            itt_max = 4500
            num_samples = 1
            num_samples_swa = 1

            num_params = K
            means = np.ones((num_params,))
            sigmas = np.ones((num_params,)) *2
            log_sigmas = np.log(sigmas)

            params = [means, sigmas]

            means_vb_clr, log_sigmas_vb_clr = means.copy(), log_sigmas.copy()
            means_vb_rms, log_sigmas_vb_rms = means.copy(), log_sigmas.copy()

            params_constant_lr = [means_vb_clr, log_sigmas_vb_clr]
            params_rms_prop = [means_vb_rms, log_sigmas_vb_rms]

            old_mean_grad = np.zeros_like(means)
            old_log_sigma_grad = np.zeros_like(log_sigmas)

            lr_t = []
            params_swa = None
            swa_n = 0
            step_size_max = 4*step_size
            step_size_min = 0.25*step_size

            lr_constant = 4e-4/N
            step_size= 4e-4/N
            step_size_rms = 0.01

            params_swa_list =[]
            params_clr_list = []
            lr_t = []
            tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
            eta = [1, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]

            l2_norm_swa = []
            l2_norm_clr = []
            l2_norm_rms = []
            #K_hat_clr_list_iter = []
            #K_hat_swa_list_iter = []
            s_mean= 0
            s_log_var=0
            start_swa_iter= 200
            elbo_threshold_swa = 0.08
            elbo_prev= 100
            elbo_diff_list = []
            elbo_list = []
            elbo_mean_list = []
            s_prev=None

            # ### Prepare sample from each of the three posterior distributions
            lr_t = np.array(lr_t)

            #la = fit_hmc.extract(permuted=True)
            #hmc_w = la['w']
            #stan_sigma = la['sigma']
            #stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta

            #params_hmc = hmc_w
            #params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
            #params_hmc_mean = np.mean(params_hmc, axis=0)
            #params_hmc_sq= np.mean(params_hmc**2, axis=0)
            #params_hmc_sigmas= np.std(params_hmc, axis=0)
            # Stan-VB
            fit_vb_samples = np.array(fit_vb['sampler_params']).T
            stan_vb_w = fit_vb_samples[:, 0:K]
        #    stan_vb_sigma = fit_vb_samples[:, 10:]

            #  VB-SWA
            #means_vb_swa = params_swa[0]
            #sigmas_vb_swa = np.exp(params_swa[1])
            #means_vb_swa = np.array(means_vb_swa)
            #samples_vb_swa = np.random.multivariate_normal(means_vb_swa, np.diag(sigmas_vb_swa), size=num_proposal_samples)
            #samples_vb_clr = np.random.multivariate_normal(means_vb_clr, np.diag(sigmas_vb_clr), size=num_proposal_samples)
            #samples_vb_rms = np.random.multivariate_normal(means_vb_rms, np.diag(sigmas_vb_rms), size=num_proposal_samples)

            #q_swa = stats.norm.pdf(samples_vb_swa, means_vb_swa, sigmas_vb_swa)
            #logp_swa = np.array([fit_hmc.log_prob(s) for s in samples_vb_swa])
            #log_iw_swa = logp_swa - np.sum(np.log(q_swa), axis=1)
            #psis_lw_swa, K_hat_swa = psislw(log_iw_swa.T)
            #print('K hat statistic for SWA')
            #print(K_hat_swa)

            #K_hat_swa_list[j,n] = K_hat_swa


            # VB-CLR

            params_vb_means = np.mean(stan_vb_w, axis=0)
            params_vb_std = np.std(stan_vb_w, axis=0)
            params_vb_sq = np.mean(stan_vb_w**2, axis=0)

            logq = stats.norm.pdf(stan_vb_w, params_vb_means, params_vb_std)
            logq_sum = np.sum(np.log(logq), axis=1)
            #log_joint_density = la['log_joint_density']
            stan_vb_log_joint_density = fit_vb_samples[:,(K+1)]
            log_iw = stan_vb_log_joint_density - logq_sum
            print(np.max(log_iw))
            print(log_iw.shape)

            psis_lw, K_hat_stan = psislw(log_iw.T)
            K_hat_stan_advi_list[j,n] = K_hat_stan
            print(psis_lw.shape)
            print('K hat statistic for Stan ADVI:')
            print(K_hat_stan)

            # l2_norm_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, W_mean, W_sigma)
            # l2_norm_hmc = np.repeat(l2_norm_hmc_i, 2200)
            #
            # l2_norm_advi_i = compute_l2_norm(means_vb_swa, np.sqrt(params_vb_std), W_mean, W_sigma)
            # l2_norm_advi = np.repeat(l2_norm_advi_i, 2200)
            # # ### Plot
            # snb.set(font_scale=1.1)
            # plt.figure(figsize=(20, 6))
            # plt.plot(l2_norm_clr[10:], label='CLR')
            # plt.plot(l2_norm_swa[10:], label='SWA')
            # plt.plot(l2_norm_rms, label='RMS')
            # plt.plot(l2_norm_hmc, label='HMC')
            # plt.plot(l2_norm_advi, label='ADVI_Stan')
            # plt.legend()
            # plt.xlabel('Likelihood Evaluations')
            # plt.ylabel('l2norm-target and current')
            # plt_title_string = 'Bayesian_Linear_Regression1_optimizers_' + str(K) + '_' + str(n)+ '.pdf'
            # plt.savefig(plt_title_string)
            # #plt.show()
            #
            # plt.figure(figsize=(20, 6))
            # plt.plot(np.array(elbo_list), label='ELBO')
            # plt.legend()
            # plt_title_string = 'Bayesian_Linear_Regression1_ELBO_' + str(K) + '_' + str(n)+ '.pdf'
            # plt.savefig(plt_title_string)
            #
            # plt.figure(figsize=(20, 6))
            # plt.plot(np.array(elbo_diff_list), label='ELBO diff')
            # plt.legend()
            # plt_title_string = 'Bayesian_Linear_Regression1_ELBO_' + str(K) + '_' + str(n)+ '.pdf'
            # plt.savefig('Bayesian_Linear_Regression_ELBO_diff_rms_20.pdf')
            #
            # plt.figure(figsize=(20, 6))
            # plt.plot(np.array(elbo_mean_list), label='ELBO mean')
            # plt.legend()
            # plt_title_string = 'Bayesian_Linear_Regression_ELBO_mean_rms_20' + str(K) + '_' + str(n)+ '.pdf'
            # plt.savefig(plt_title_string)
            # plt.show()

        # plt.figure(figsize=(20, 6))
        # plt.plot(K_hat_clr_list[10:], label='CLR')
        # plt.plot(K_hat_swa_list[10:], label='SWA')
        # plt.legend()
        # plt.savefig('Bayesian_Linear_Regression_K_hat1_50_2.pdf')
        # plt.show()


        print('hii')
        #plt.figure(figsize=(20,6))

#plt.scatter(K_list, np.mean(K_hat_rms_list, axis=1), 'g-')
#plt.scatter(K_list, np.mean(K_hat_swa_list, axis=1), 'r-')
#plt.scatter(K_list, np.mean(K_hat_clr_list, axis=1), 'm-')
plt.plot(K_list, np.mean(K_hat_stan_advi_list, axis=1), 'r-', alpha=1)
plt.plot(K_list, np.min(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.max(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.ylim((0,5))

plt.legend()
plt.savefig('Linear_Regression_K_hat_vs_D.pdf')
#plt.show()
