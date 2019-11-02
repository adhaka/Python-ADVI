
'''
file to plot K hat values vs D where D is the dimensionality of the variational parameters for a
linear gaussian model where the covariates are correlated, and mean field approximation is
insufficient but we still use it just for illustration purposes.
'''

import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from GPy.util import choleskies

import GPy
import numpy
numpy.set_printoptions(edgeitems=3,infstr='inf',linewidth=75, nanstr='nan', precision=4)
import autograd.numpy as np
import seaborn as snb
import pystan
import scipy
from scipy import stats
import pickle
#import scipy.stats as stats
import argparse

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule, stepsize_linear_weight_averaging, stepsize_linear_adaptive_is_mixing_schedule, rms_prop_gradient
from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm
from autograd import grad
from arviz import psislw


np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument('--N', '-n', type=int, default=1000)

args = parser.parse_args()
N_user = args.N

##  code for linear model with fixed variances .
linear_regression_code= """
data{
    int<lower=0> N;
    int<lower=0> K;
    matrix[N,K] X;
    vector[N] y;
    real<lower=0> sigma;
}

parameters{
vector[K] w;
}

model{
w ~ normal(4,0.5);
y ~ normal(X*w , sigma);
}

generated quantities{
real log_joint_density;
log_joint_density = normal_lpdf(y|X*w, sigma) + normal_lpdf(w| 0, 1) + log(sigma);
}
"""

np.random.seed(123)

logit = lambda x: 1./ (1 +np.exp(-x))
# Some helper functions

def compute_entropy(L, Sigma=None):
    return np.sum(np.log(np.diag(L))) + L.shape[0]*(1 + np.log(2*np.pi))


def reparametrize(zs, means, L):
    tmp = np.dot(L, zs)
    #print(means[:,None])
    samples = means[:, None] + tmp
    Linv = np.linalg.inv(L)
    grad_correction = zs.T
    return samples, grad_correction



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



N_train = N_user
N = N_user
K=  2
K_list = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
K_list = [5, 10, 19, 30, 40,50]

num_K = len(K_list)
M = 1
w0= 0
N_sim = 2
K_hat_stan_advi_list = np.zeros((num_K, N_sim))
debug_mode = True

for j in range(num_K):
    for n in range(N_sim):
        mean = np.zeros((K,))
        cov = np.ones((K,K))
        K = K_list[j]

        #cov = np.asarray([[1, 0.5 ], [0.5, 1]])
        #cov = GPy.kern.RBF(X)
        cov_inverted = np.asarray([[1, -0.5 ], [-0.5, 1]])
        X_tmp = np.random.normal(0, 1, (N,K))
        X=X_tmp

        Z= np.linspace(-1,1, K)
        Z = Z[:,np.newaxis]
        rbf_kernel = GPy.kern.RBF(lengthscale=1, input_dim=1)
        covar= rbf_kernel.K(Z)

        #X = np.concatenate((X_tmp, np.ones((N,1))), axis=1)

        #Y = np.dot(X, W)
        print(X.shape)
        w_mean = 0
        w_mean = 4
        w_sigma= 0.5
        #w_sigma = 1
        w_mean_true = w_mean
        w_sigma_true = w_sigma

        W_mean = np.ones((K, ))*2
        W_cov = covar
        #W_mean = np.asarray([2, 2])
        #W_cov = np.asarray([[0.99,0.004],[0.004, 0.99]])

        # W= W_mean
        print(W_mean.shape)
        print(W_cov.shape)
        W = np.random.multivariate_normal(W_mean, W_cov, 1).T
        print(W.shape)
        #W = np.random.normal(w_mean, w_sigma, (K,M))
        #W = np.concatenate((W, np.ones((1,M))*w0), axis=0)

        W_mean = np.repeat(w_mean, K)
        W_sigma = np.repeat(w_sigma, K)

        #sigma_0 = np.random.gamma(0.5, 0.5, M)
        sigma_0 = 0.90
        sigma_0 = 0.25
        y_mean= X@W
        Y = y_mean + np.random.normal(0, sigma_0, (N,M))

        #M is the number of simulations for the whole model.
        for i in range(1,M+1):
            model_data= {'N':N_train,
               'K':K,
               'y':Y[:,0],
               'X':X,
               'sigma':sigma_0
               }

            #sm = pystan.StanModel(model_code=linear_regression_code)
            try:
                sm = pickle.load(open('model_correlation_regression_10.pkl', 'rb'))
            except:
                sm = pystan.StanModel(model_code=linear_regression_code)
                with open('model_correlation_regression_10.pkl', 'wb') as f:
                    pickle.dump(sm, f)

            num_proposal_samples = 6000
            #fit_hmc = sm.sampling(data=model_data, iter=800)
            fit_vb = sm.vb(data=model_data, iter=30000, tol_rel_obj=1e-4, output_samples=num_proposal_samples,
                           algorithm='meanfield')
            # ### Run ADVI in Python
            # use analytical gradient of entropy
            compute_entropy_grad = grad(compute_entropy)

            # settings
            step_size = 4e-7
            step_size= 4e-2/N_train
            itt_max = 3000
            num_samples = 1

            num_params = K
            means_vb = np.ones((num_params,))
            betas_vb = np.ones((num_params*(num_params+1)//2,1))

            tmpL= np.tril(np.ones((K,K)))
            #C = np.ones((K,K))
            L_vb =  choleskies._flat_to_triang_pure(betas_vb)[0,:]
            Sigma = L_vb@L_vb.T

            params = [means_vb, betas_vb]
            means_vb_clr, betas_vb_clr = means_vb.copy(), betas_vb.copy()
            means_vb_swa, betas_vb_swa = means_vb.copy(), betas_vb.copy()
            means_vb_rms, betas_vb_rms = means_vb.copy(), betas_vb.copy()
            L_vb_clr = L_vb.copy()
            L_vb_swa = L_vb.copy()
            L_vb_rms = L_vb.copy()

            params_constant_lr = [means_vb_clr, betas_vb_clr]
            params_rms = [means_vb_rms, betas_vb_rms]

            old_mean_grad = np.zeros_like(means_vb)
            old_beta_grad = np.zeros_like(betas_vb)
            old_L_grad = np.zeros_like(L_vb)
            lr_t = []
            params_swa = None
            swa_n = 0
            step_size_max = 1.5*step_size
            step_size_min = 0.95*step_size

            lr_constant = 2e-2/N_train
            step_size= 2e-2/N_train
            step_size_rms = 0.2

            params_swa_list =[]
            params_clr_list = []
            lr_t = []
            tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
            eta = [1, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]

            l2_norm_swa = []
            l2_norm_clr = []
            l2_norm_rms = []
            K_hat_clr_list = []
            K_hat_swa_list = []
            K_hat_rms_list = []
            s_mean= 0
            s_log_var=0
            start_swa_iter= 800
            start_swa = False
            elbo_threshold_swa = 0.15
            elbo_prev= 100
            elbo_diff_list = []
            elbo_list = []
            elbo_mean_list = []
            s_prev= None
            densities = dict()
            densities['sd'] = 0.
            densities['sum'] = 0.
            sum_density = 0.
            is_correction = True
            num_samples_swa = 1


            for itt in range(itt_max):
                zs = np.random.normal(0, 1, size=(num_params, num_samples))
                zs_swa = np.random.normal(0, 1, size=(num_params, num_samples_swa))
                if itt == start_swa_iter:
                    print(L_vb_swa)
                    print(means_vb_swa)


                #mean_grads_running_dot_product = np.mean(mean_grad*old_mean_grad)
                #sigma_grads_running_dot_product = np.mean(log_sigma_grad*old_log_sigma_grad)
                #s_mean += mean_grads_running_dot_product
                #s_log_var += sigma_grads_running_dot_product
                #elbo_diff_list.append(elbo - elbo_prev)
                #elbo_diff_median =  np.median(np.array(elbo_diff_list[-21:-1]))
                #elbo_diff_mean = np.mean(np.array(elbo_diff_list[-21:-1]))
                #elbo_diff_last_20 = elbo_diff_list[-20:-1]
                #elbo_diff_max = np.max(np.array(elbo_diff_list[-21:-1]))
                #mean_grad = mean_grad[:, np.newaxis]

                #densities['sd'] = 1.

                #ensities['sd'] = np.mean(samples_p_density, axis=0)/np.mean(samples_q_density)
                #densities['sd'] = mean_q_density
                #densities['sd'] = samples_q_density
                #params = params_swa

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

            #stan_sigma = la['sigma']
            #stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta


            # Stan-VB
            fit_vb_samples = np.array(fit_vb['sampler_params']).T
            print(fit_vb_samples[1,:])
            stan_vb_w = fit_vb_samples[:,:K]
            stan_vb_mean = np.mean(stan_vb_w, axis=0)
            stan_vb_cov = np.cov(stan_vb_w[:,0], stan_vb_w[:,1])

        #    stan_vb_sigma = fit_vb_samples[:, 10:]


            params_vb_means = np.mean(stan_vb_w, axis=0)
            params_vb_std = np.std(stan_vb_w, axis=0)
            params_vb_sq = np.mean(stan_vb_w**2, axis=0)

            logq = stats.norm.pdf(stan_vb_w, params_vb_means, params_vb_std)
            logq_sum = np.sum(np.log(logq), axis=1)
            # log_joint_density = la['log_joint_density']
            stan_vb_log_joint_density = fit_vb_samples[:, K]
            log_iw = stan_vb_log_joint_density - logq_sum
            print(np.max(log_iw))
            print(log_iw.shape)

            psis_lw, K_hat_stan = psislw(log_iw.T)
            K_hat_stan_advi_list[j, n] = K_hat_stan
            print(psis_lw.shape)
            print('K hat statistic for Stan ADVI:')
            print(K_hat_stan)


    ###################### Plotting L2 norm here #################################

plt.figure()
plt.plot(stan_vb_w[:,0], stan_vb_w[:,1], 'mo', label='STAN-ADVI')
plt.savefig('vb_w_samples_mf.pdf')


            # plt.figure(figsize=(20, 6))
            # plt.plot(K_hat_clr_list[10:], label='CLR')
            # plt.plot(K_hat_swa_list[10:], label='SWA')
            # plt.legend()
            # plt.savefig('Bayesian_Linear_Regression_K_hat1_50_2.pdf')
            # plt.show()

plt.figure()
plt.plot(K_list, np.mean(K_hat_stan_advi_list, axis=1), 'r-', alpha=1)
plt.plot(K_list, np.min(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.max(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.ylim((0,5))

plt.legend()
plt.savefig('Linear_Regression_K_hat_vs_D_mf_5000N.pdf')
