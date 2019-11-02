
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

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule, stepsize_linear_weight_averaging, stepsize_linear_adaptive_is_mixing_schedule, rms_prop_gradient, step_size_rms_prop_schedule
from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm, compute_l2_norm_means, compute_l2_norm_Sigmas, \
    compute_R_hat_fn
from autograd import grad
from arviz import psislw
from swa_schedules import scale_factor_warm_up
from swa_schedules import elbo_logit, elbo_grad_logit


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

np.random.seed(162)
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

N_train = 1000
N = 1000
K_list = [10]
num_K = len(K_list)
M = 1
w0= 0
N_sim = 4

debug_mode = True
compute_L2_norm = True

K_hat_stan_advi_list = np.zeros((num_K, N_sim))
l2_norm_rms_list = []

itt_max = 35000

compute_R_hat = True



for j in range(num_K):
    K= K_list[j]
    vi_params_collapsed = np.zeros((N_sim, itt_max, 2*K))
    vi_params_collapsed2 = np.zeros((N_sim, itt_max, 2*K))
    means_all = np.zeros((K * N_sim))
    sigmas_all = np.ones((K * N_sim))


    for n in range(N_sim):
        N_train = 1000
        N_test = 50
        K = K_list[j]
        M = 1
        w0 = 0

        mean = np.zeros((K,))
        cov = np.ones((K, K))

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

        w_mean = 0
        w_mean = 5
        w_sigma= 0.2
        #w_sigma = 1


        #W_mean = np.ones((K,))*4
        #W_cov = covar
        #W = np.random.multivariate_normal(W_mean, W_cov, 1).T
        w_mean = 0
        w_mean = 5
        w_sigma= 0.5
        #w_sigma = 1
        w_mean_true = w_mean
        w_sigma_true = w_sigma

        W = np.random.normal(w_mean, w_sigma, (K,M))

        W_mean = np.repeat(w_mean, K)
        W_sigma = np.repeat(w_sigma, K)

        W_cov = W_sigma
        y_mean = x_full @ W
        p_full = logit(y_mean)
        y_full = np.random.binomial(n=1, p=p_full)
        Y = y_full[:N_train]
        Y_test = y_full[N_train:]

        model_data = {'N': N_train,
                      'K': K,
                      'y': Y[:, 0],
                      'X': X
                      }

        try:
            sm = pickle.load(open('model_logistic.pkl', 'rb'))
        except:
            sm = pystan.StanModel(model_code=linear_reg_fixed_variance_code)
            with open('model_logistic.pkl', 'wb') as f:
                pickle.dump(sm, f)


        num_proposal_samples = 7000
        fit_hmc = sm.sampling(data=model_data, iter=2000)
        fit_vb = sm.vb(data=model_data, iter=30000, tol_rel_obj=1e-5, output_samples=num_proposal_samples)
        # ### Run ADVI in Python
        # use analytical gradient of entropy
        compute_entropy_grad = grad(compute_entropy)

        # settings
        #step_size= 1e-5/N
        #itt_max = 30000
        num_samples = 1
        num_samples_swa = 1

        num_params = K
        means1 =  np.ones((num_params,))*4
        means2 = np.zeros((num_params,))


        means = means_all[n*num_params:(n+1)*num_params]
        sigmas = sigmas_all[n*num_params:(n+1)*num_params]

        means_list = [means1, means2]

        sigmas1 = np.ones((num_params,)) *2
        sigmas2 = np.ones((num_params,))*4
        sigmas_list = [sigmas1, sigmas2]

        #means = means_list[n]
        #sigmas =  sigmas_list[n]
        params = [means.copy(), sigmas.copy()]
        log_sigmas = np.log(sigmas)
        means_vb_clr, log_sigmas_vb_clr = means.copy(), log_sigmas.copy()
        means_vb_rms, log_sigmas_vb_rms = means.copy(), log_sigmas.copy()
        means_vb_clr2, log_sigmas_vb_clr2 = means.copy(), log_sigmas.copy()
        params_constant_lr = [means_vb_clr, log_sigmas_vb_clr]
        params_constant_lr2 = [means_vb_clr2, log_sigmas_vb_clr2]
        params_rms_prop = [means_vb_rms, log_sigmas_vb_rms]
        old_mean_grad = np.zeros_like(means)
        old_log_sigma_grad = np.zeros_like(log_sigmas)
        #lr_t = []
        params_swa = None
        swa_n = 0
        start_swa = False
        step_size_clr = 4e-3/N
        step_size= 4e-4/N
        step_size_rms = 0.00005
        params_swa_list =[]
        params_clr_list = []
        lr_t = np.zeros((itt_max,1))
        tol_vec = [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-8]
        eta = [1, 0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005]

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

        scale_factor= scale_factor_warm_up(X, Y)
        step_size = scale_factor/10
        step_size_rms = scale_factor/10

        step_size_max = 1.05*step_size
        step_size_min = 0.995*step_size

        for itt in range(itt_max):
            zs = np.random.normal(0, 1, size=(num_params, num_samples))
            zs_swa = np.random.normal(0, 1, size=(num_params, num_samples_swa))

            dof = 4
            zs_t = np.random.standard_t(dof, size=(num_params, num_samples))
            zs_t_swa = np.random.standard_t(dof, size=(num_params, num_samples_swa))
            samples, grad_correction = reparametrize(zs_swa, means, log_sigmas)
            samples_clr, grad_correction_clr = reparametrize(zs, means_vb_clr, log_sigmas_vb_clr)
            samples_rms, grad_correction_rms = reparametrize(zs, means_vb_rms, log_sigmas_vb_rms)
            samples_clr2, grad_correction_clr2 = reparametrize(zs, means_vb_clr2, log_sigmas_vb_clr2)
            #samples_swa2, grad_correction_swa2 = reparametrize(zs, means_vb_clr2, log_sigmas_vb_clr2)
            #samples, grad_correction = reparametrize(zs_swa, means, log_sigmas)

            #log_p_grad1 = np.array([fit_hmc.grad_log_prob(s)])
            # evaluate gradient of log p (does grad_log_prob support vectorization??) and gradient of log q
            log_p_grad = np.array([fit_hmc.grad_log_prob(s) for s in samples.T])
            log_p_grad_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])
            log_p_grad_rms = np.array([fit_hmc.grad_log_prob(s) for s in samples_rms.T])
            log_p_grad_clr2 = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr2.T])

            # using analytical gradients
            log_p_grad_analytical = elbo_grad_logit(zs_swa, means, log_sigmas, X, Y)
            log_p_grad_clr_analytical = elbo_grad_logit(zs, means_vb_clr, log_sigmas_vb_clr, X, Y)
            log_p_grad_rms_analytical = elbo_grad_logit(zs, means_vb_rms, log_sigmas_vb_rms, X, Y)
            log_p_grad_clr2_analytical = elbo_grad_logit(zs, means_vb_clr2, log_sigmas_vb_clr2, X, Y)

            mean_grad_analytical = log_p_grad_analytical[:K]
            log_sigma_grad_analytical = log_p_grad_analytical[K:]

            mean_grad_rms_analytical = log_p_grad_rms_analytical[:K]
            log_sigma_grad_rms_analytical = log_p_grad_rms_analytical[K:]

            mean_grad_clr2_analytical = log_p_grad_clr2_analytical[:K]
            log_sigma_grad_clr2_analytical = log_p_grad_clr2_analytical[K:]

            mean_grad_clr_analytical = log_p_grad_clr_analytical[:K]
            log_sigma_grad_clr_analytical = log_p_grad_clr_analytical[K:]
            #log_p_grad = log_p_grad_analytical
            #log_p_grad_clr = log_p_grad_clr_analytical
            #log_p_grad_rms = log_p_grad_rms_analytical
            #log_p_grad_clr2 = log_p_grad_clr2_analytical

            log_p = np.array([fit_hmc.log_prob(s) for s in samples.T])
            elbo= np.mean(log_p) + compute_entropy(log_sigmas)
            elbo_analytical = elbo_logit(zs, means, log_sigmas, X, Y)

            log_p_clr = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr.T])
            elbo_clr = np.mean(log_p_clr) + compute_entropy(log_sigmas_vb_clr)
            elbo_clr_analytical = elbo_logit(zs, means_vb_clr, log_sigmas_vb_clr, X, Y)

            log_p_clr2 = np.array([fit_hmc.grad_log_prob(s) for s in samples_clr2.T])
            elbo_clr2 = np.mean(log_p_clr2) + compute_entropy(log_sigmas_vb_clr2)
            elbo_clr2_analytical = elbo_logit(zs, means_vb_clr2, log_sigmas_vb_clr2, X, Y)

            log_p_rms = np.array([fit_hmc.grad_log_prob(s) for s in samples_rms.T])
            elbo_rms = np.mean(log_p_rms) + compute_entropy(log_sigmas_vb_rms)
            elbo_rms_analytical = elbo_logit(zs, means_vb_rms, log_sigmas_vb_rms, X, Y)

            entropy_grad = compute_entropy_grad(log_sigmas)
            entropy_grad_clr = compute_entropy_grad(log_sigmas_vb_clr)
            entropy_grad_rms = compute_entropy_grad(log_sigmas_vb_rms)
            entropy_grad_clr2 = compute_entropy_grad(log_sigmas_vb_clr2)
            # compute gradients wrt. mean and log_sigma
            mean_grad = np.mean(log_p_grad, axis=0)
            log_sigma_grad =np.mean(log_p_grad*grad_correction, axis=0) + entropy_grad

            la = fit_hmc.extract(permuted=True)
            hmc_w = la['w']
            # stan_sigma = la['sigma']
            # stan_theta = stan_mu[:, None] + stan_tau[:, None]*stan_eta

            params_hmc = hmc_w
            # params_stan = np.concatenate((stan_w, stan_sigma), axis=0)
            params_hmc_mean = np.mean(params_hmc, axis=0)
            params_hmc_sq = np.mean(params_hmc ** 2, axis=0)
            params_hmc_sigmas = np.std(params_hmc, axis=0)

            # Inner product of gradients
            #print('mean gradient')
            #print(mean_grad)
            #print('old mean gradient')
            #print(old_mean_grad)

            mean_grads_running_dot_product = np.mean(mean_grad*old_mean_grad)
            sigma_grads_running_dot_product = np.mean(log_sigma_grad*old_log_sigma_grad)

            s_mean += mean_grads_running_dot_product
            s_log_var += sigma_grads_running_dot_product
            old_mean_grad = mean_grad
            old_log_sigma_grad = log_sigma_grad

            #if (itt+1) % 10== 0:
                #print(mean_grads_running_dot_product)
                #print(sigma_grads_running_dot_product)


            criterion1 = mean_grads_running_dot_product < 0
            criterion2 = sigma_grads_running_dot_product < 0
            criterion3 = np.abs(elbo_prev - elbo) < np.abs(elbo_threshold_swa*elbo_prev)
            criterion7 =  s_mean < 0
            criterion8 = s_log_var < 0
            elbo_diff_list.append(elbo - elbo_prev)

            elbo_diff_median =  np.median(np.array(elbo_diff_list[-21:-1]))
            elbo_diff_mean = np.mean(np.array(elbo_diff_list[-21:-1]))
            elbo_diff_last_20 = elbo_diff_list[-20:-1]
            #elbo_diff_max = np.max(np.array(elbo_diff_list[-21:-1]))
            elbo_diff_list_abs = [0 for i in elbo_diff_last_20 if i < 0]
            val1 = len(elbo_diff_list_abs) - np.count_nonzero(np.asarray(elbo_diff_list_abs))
            criterion4 = val1 > 5

            criterion6 = itt > 4000
            if len(elbo_mean_list) > 6:
                criterion5 = np.abs(elbo_mean_list[-1] - elbo_mean_list[-2]) < np.abs(elbo_mean_list[-2] - elbo_mean_list[-5])*0.10

            #if criterion1 and criterion2 and criterion3 and criterion6 and start_swa is False:
            #    start_swa = True
            #    start_swa_iter = itt+1
            #    print(start_swa_iter)
            #    num_samples_swa =4
                #print(elbo_diff_list)
                #exit()

            if criterion1 and criterion2 and criterion6 and start_swa is False:
                start_swa = True
                start_swa_iter = itt+1
                print(start_swa_iter)
                num_samples_swa = 3
                #exit()

            #mean_grad = mean_grad[:, np.newaxis]
            mean_grad_clr = np.mean(log_p_grad_clr, axis=0)
            log_sigma_grad_clr = np.mean(log_p_grad_clr*grad_correction_clr, axis=0) + entropy_grad_clr

            mean_grad_clr2 = np.mean(log_p_grad_clr2, axis=0)
            log_sigma_grad_clr2 = np.mean(log_p_grad_clr2*grad_correction_clr2, axis=0) + entropy_grad_clr2
            mean_grad_rms = np.mean(log_p_grad_rms, axis=0)
            log_sigma_grad_rms =np.mean(log_p_grad_rms*grad_correction_rms, axis=0) + entropy_grad_rms
            # take gradient step
            print(step_size)
            #means_vb_clr += step_size*mean_grad_clr
            #log_sigmas_vb_clr += step_size*log_sigma_grad_clr
            #means_vb_clr += step_size*mean_grad_clr
            #log_sigmas_vb_clr += step_size*log_sigma_grad_clr
            means_vb_clr += step_size_clr*mean_grad_clr
            log_sigmas_vb_clr += step_size_clr*log_sigma_grad_clr

            means_vb_clr2 += step_size*mean_grad_clr2
            log_sigmas_vb_clr2 += step_size*log_sigma_grad_clr2
            #print(means)
            #print(log_sigmas)
            params = [means, log_sigmas]
            params_rms_prop = [means_vb_rms, log_sigmas_vb_rms]
            #step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
            #itt+1, itt_max+1, start_swa_iter, 80, params_swa, swa_n)
            step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
            itt+1, itt_max+1, start_swa_iter, 200, params_swa, swa_n)
            rho, s, params_rms_prop = step_size_rms_prop_schedule(params_rms_prop,
                    itt+1, mean_grad_rms, log_sigma_grad_rms, s_prev, step_size_rms)
            s_prev = s
            #step_size_min = step_size
            #step_size_max = step_size+1e-10
            #step_size, params_swa, swa_n = stepsize_linear_adaptive_schedule(params, step_size, step_size_min, step_size_max,
            #itt+1, itt_max+1, 2, 1, params_swa, swa_n)
            means += step_size*mean_grad
            log_sigmas += step_size*log_sigma_grad

            means_vb_rms = params_rms_prop[0]
            log_sigmas_vb_rms = params_rms_prop[1]
            #params = params_swa
            lr_t[itt] = step_size
            #  transform back to constrained space
            sigmas = np.exp(log_sigmas)
            sigmas_vb_clr = np.exp(log_sigmas_vb_clr.copy())
            sigmas_vb_clr2 = np.exp(log_sigmas_vb_clr2.copy())
            sigmas_vb_rms = np.exp(log_sigmas_vb_rms.copy())
            params_vb_clr  = [means_vb_clr, sigmas_vb_clr]
            params_vb_clr2 = [means_vb_clr2, sigmas_vb_clr2]
            params_rms_prop = [means_vb_rms, log_sigmas_vb_rms]
            #params_clr_list.append(params_vb_clr)
            #params_swa_list.append(params_vb_swa2)
            means_vb_swa = params_swa[0].copy()
            sigmas_vb_swa = np.exp(params_swa[1]).copy()


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
                l2_norm_clr_i = compute_l2_norm(W_mean, W_cov, means_vb_clr, sigmas_vb_clr)
                l2_norm_swa_i = compute_l2_norm(W_mean, W_cov, means_vb_swa, sigmas_vb_swa)
                l2_norm_rms_i = compute_l2_norm(W_mean, W_cov, means_vb_rms, sigmas_vb_rms)
                l2_norm_clr2_i = compute_l2_norm(W_mean, W_cov, means_vb_clr2, sigmas_vb_clr2)

                l2_norm_clr_means_i = compute_l2_norm_means(W_mean, means_vb_clr)
                l2_norm_clr_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_clr)
                l2_norm_swa_means_i = compute_l2_norm_means(W_mean, means_vb_swa)
                l2_norm_swa_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_swa)
                l2_norm_rms_means_i = compute_l2_norm_means(W_mean, means_vb_rms)
                l2_norm_rms_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_rms)
                l2_norm_clr2_means_i = compute_l2_norm_means(W_mean, means_vb_clr2)
                l2_norm_clr2_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_clr2)

                l2_norm_clr_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_clr, sigmas_vb_clr)
                l2_norm_swa_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_swa, sigmas_vb_swa)
                l2_norm_rms_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_rms, sigmas_vb_rms)
                l2_norm_clr2_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, means_vb_clr2, sigmas_vb_clr2)

                l2_norm_clr_means_i = compute_l2_norm_means(W_mean, means_vb_clr)
                l2_norm_clr_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_clr)
                l2_norm_swa_means_i = compute_l2_norm_means(W_mean, means_vb_swa)
                l2_norm_swa_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_swa)
                l2_norm_rms_means_i = compute_l2_norm_means(W_mean, means_vb_rms)
                l2_norm_rms_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_rms)
                l2_norm_clr2_means_i = compute_l2_norm_means(W_mean, means_vb_clr2)
                l2_norm_clr2_sigmas_i = compute_l2_norm_Sigmas(W_cov, sigmas_vb_clr2)

                l2_norm_hmc_clr_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_clr)
                l2_norm_hmc_clr_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, sigmas_vb_clr)
                l2_norm_hmc_swa_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_swa)
                l2_norm_hmc_swa_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, sigmas_vb_swa)
                l2_norm_hmc_rms_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_rms)
                l2_norm_hmc_rms_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, sigmas_vb_rms)
                l2_norm_hmc_clr2_means_i = compute_l2_norm_means(params_hmc_mean, means_vb_clr2)
                l2_norm_hmc_clr2_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, sigmas_vb_clr2)

                l2_norm_means_hmc_swa.append(l2_norm_hmc_swa_means_i)
                l2_norm_means_hmc_clr.append(l2_norm_hmc_clr_means_i)
                l2_norm_means_hmc_rms.append(l2_norm_hmc_rms_means_i)
                l2_norm_means_hmc_clr2.append(l2_norm_hmc_clr2_means_i)

                l2_norm_sigmas_hmc_swa.append(l2_norm_hmc_swa_means_i)
                l2_norm_sigmas_hmc_clr.append(l2_norm_hmc_clr_means_i)
                l2_norm_sigmas_hmc_rms.append(l2_norm_hmc_rms_means_i)
                l2_norm_sigmas_hmc_clr2.append(l2_norm_hmc_clr2_means_i)

                l2_norm_means_swa.append(l2_norm_hmc_swa_means_i)
                l2_norm_means_clr.append(l2_norm_hmc_clr_means_i)
                l2_norm_means_rms.append(l2_norm_hmc_rms_means_i)
                l2_norm_means_clr2.append(l2_norm_hmc_clr2_means_i)

                l2_norm_sigmas_swa.append(l2_norm_swa_sigmas_i)
                l2_norm_sigmas_clr.append(l2_norm_clr_sigmas_i)
                l2_norm_sigmas_rms.append(l2_norm_rms_sigmas_i)
                l2_norm_sigmas_clr2.append(l2_norm_clr2_sigmas_i)

                l2_norm_swa.append(l2_norm_swa_i)
                l2_norm_clr.append(l2_norm_clr_i)
                l2_norm_rms.append(l2_norm_rms_i)
                l2_norm_clr2.append(l2_norm_clr2_i)

                l2_norm_hmc_swa.append(l2_norm_swa_hmc_i)
                l2_norm_hmc_clr.append(l2_norm_clr_hmc_i)
                l2_norm_hmc_rms.append(l2_norm_rms_hmc_i)
                l2_norm_hmc_clr2.append(l2_norm_clr2_hmc_i)


            elbo_prev= elbo
            elbo_list.append(elbo)
            elbo_mean_list.append(np.mean(elbo_list[-20:-1]))


            if compute_R_hat:
                #print(vi_params_collapsed.shape)
                #a1 = np.array([means_vb_clr.ravel(), sigmas_vb_clr.ravel()])
                a2 = np.concatenate((means_vb_clr2.flatten(), sigmas_vb_clr2.flatten()), axis=None)
                a1 = np.concatenate((means_vb_clr.flatten(), sigmas_vb_clr.flatten()), axis=None)
                #print(a1)
                vi_params_collapsed[n, itt] = a1
                vi_params_collapsed2[n, itt] = a2

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
            #
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

fit_vb_samples = np.array(fit_vb['sampler_params']).T
stan_vb_w = fit_vb_samples[:, 0:K]
params_vb_stan_means = np.mean(stan_vb_w, axis=0)
params_vb_stan_std = np.std(stan_vb_w, axis=0)
params_vb_stan_sq = np.mean(stan_vb_w ** 2, axis=0)

l2_norm_vb_stan_i = compute_l2_norm(params_vb_stan_means, params_vb_stan_std, W_mean, np.diag(W_cov))
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

l2_norm_hmc_i = compute_l2_norm(params_hmc_mean, params_hmc_sigmas, W_mean, np.diag(W_cov))
l2_norm_hmc = np.repeat(l2_norm_hmc_i, itt_max)

l2_norm_hmc_means_i = compute_l2_norm_means(params_hmc_mean, W_mean)
l2_norm_hmc_sigmas_i = compute_l2_norm_Sigmas(params_hmc_sigmas, W_cov)
l2_norm_hmc_means = np.repeat(l2_norm_hmc_means_i, itt_max)
l2_norm_hmc_sigmas = np.repeat(l2_norm_hmc_sigmas_i, itt_max)

l2_norm_vb_stan_means_i = compute_l2_norm_means(params_vb_stan_means, W_mean)
l2_norm_vb_stan_sigmas_i = compute_l2_norm_Sigmas(params_vb_stan_std, W_cov)
l2_norm_vb_stan_means = np.repeat(l2_norm_vb_stan_means_i, itt_max)
l2_norm_vb_stan_sigmas = np.repeat(l2_norm_vb_stan_sigmas_i, itt_max)

warmup = start_swa_iter
if start_swa_iter %2 == 1:
    warmup = warmup +1

if warmup < 3000:
    warmup = 3000


chains = vi_params_collapsed
chains_clr2 = vi_params_collapsed2
chains2 = vi_params_collapsed[:,warmup:,:]

R_hat = compute_R_hat_fn(chains, warmup)
print(R_hat)
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
    print(sigmas_vb_clr)
    print(means_vb_clr2)
    print(sigmas_vb_clr2)
    print(means_vb_rms)
    print(sigmas_vb_rms)
    print(means_vb_swa)
    print(sigmas_vb_swa)
    print(vi_params_collapsed.shape)
    #exit()

#plt.scatter(K_list, np.mean(K_hat_rms_list, axis=1), 'g-')
#plt.scatter(K_list, np.mean(K_hat_swa_list, axis=1), 'r-')
#plt.scatter(K_list, np.mean(K_hat_clr_list, axis=1), 'm-')
#plt.plot(K_list, np.mean(K_hat_stan_advi_list, axis=1), 'y-')
plt.figure()
plt.plot(l2_norm_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_swa, 'b-', label='SWA')
plt.plot(l2_norm_clr2, 'm-', label='CLR2')
plt.plot(l2_norm_hmc, 'y-', label='HMC')
plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_10D_500N_Gelbo.pdf')



plt.figure()
plt.plot(l2_norm_means_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_means_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_means_swa, 'b-', label='SWA')
plt.plot(l2_norm_means_clr2, 'm-', label='CLR2')
#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_mean_10D_500N_Gelbo.pdf')


plt.figure()
plt.plot(l2_norm_sigmas_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_sigmas_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_sigmas_swa, 'b-', label='SWA')
plt.plot(l2_norm_sigmas_clr2, 'm-', label='CLR2')
#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_sigmas_10D_500N_Gelbo.pdf')




plt.figure()
plt.plot(l2_norm_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_hmc_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_hmc_swa, 'b-', label='SWA')
plt.plot(l2_norm_hmc_clr2, 'm-', label='CLR2')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_10D_500N_hmc_Gelbo.pdf')



plt.figure()
plt.plot(l2_norm_means_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_means_hmc_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_means_hmc_swa, 'b-', label='SWA')
plt.plot(l2_norm_means_hmc_clr2, 'm-', label='CLR2')
#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_mean_10D_500N_hmc_Gelbo.pdf')
#plt.show()

plt.figure()
plt.plot(l2_norm_sigmas_hmc_rms, 'r-', label='RMS_Prop')
plt.plot(l2_norm_sigmas_hmc_clr, 'g-', label='CLR_RMS')
plt.plot(l2_norm_sigmas_hmc_swa, 'b-', label='SWA')
plt.plot(l2_norm_sigmas_hmc_clr2, 'm-', label='CLR2')
#plt.plot(l2_norm_means_hmc, 'y-', label='HMC')
#plt.plot(l2_norm_vb, '-', label='VB')
plt.yscale('log')
plt.legend()
plt.savefig('Logistic_Regression_optimizers_sigmas_10D_500N_hmc_Gelbo.pdf')
