
'''
file to plot K hat values vs D where D is the dimensionality of the variational parameters for a
linear gaussian model where the covariates are correlated, and mean field approximation is
insufficient and hence we use full rank approximation here.

'''


import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

import sys
sys.path.append('..')

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

from swa_schedules import stepsize_cyclical_adaptive_schedule, stepsize_linear_adaptive_schedule

from helper import compute_entropy, gaussian_entropy, expectation_iw, compute_l2_norm
from autograd import grad
from arviz import psislw
from data_generator import data_generator_linear, data_generator_logistic
from functions import add_noise, scale_factor_warm_up, elbo_grad_gaussian, elbo_gaussian, elbo_full

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-a', type=int, default=1)
parser.add_argument('--N', '-n', type=int, default=1000)
parser.add_argument('--iters', '-i', type=int, default=80000)
parser.add_argument('--samplesgrad', '-s', type=int, default=1)
parser.add_argument('--sampleselbo', '-e', type=int, default=1)
parser.add_argument('--evalelbo', '-l', type=int, default=100)
parser.add_argument('--datatype', '-d', type=str, default='independent')

args = parser.parse_args()

max_iters = args.iters
algo = 'meanfield'
algo_name='mf'
gradsamples = args.samplesgrad
elbosamples = args.sampleselbo
evalelbo = args.evalelbo
datatype= args.datatype


if args.algorithm ==1:
    algo = 'meanfield'
    algo_name = 'mf'
elif args.algorithm ==2:
    algo = 'fullrank'
    algo_name = 'fr'

np.set_printoptions(precision=3)
## code for general linear model without any constraints and gamma prior for std dev.
##  code for linear model with fixed variances .
logistic_reg_fixed_variance_code= """
functions{

#    vector logit(vector x){
#        real t;
#        t = 1. /(1 + exp(-x));
#        return t;
#    }


}


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

SEED= 2019
logit = lambda x: 1./ (1 +np.exp(-x))
np.random.seed(SEED)
N =args.N

try:
    a1 = np.load('K_hat_logistic_'+ datatype + '_' + algo_name + '_' + str(N) + 'N.npy')
except:
    pass

K=  2
K_list = [5, 10, 20, 30, 40, 50, 60]
#K_list = [5, 10, 19, 30, 40,50]

num_K = len(K_list)
M = 1
w0= 0
N_sim = 6
K_hat_stan_advi_list = np.zeros((num_K, N_sim))
debug_mode = True
N_test = 50
noise_sigma = 1.0
noise_var = noise_sigma**2

for j in range(num_K):
    for n in range(N_sim):
        K = K_list[j]
        regression_data= data_generator_logistic(N, K, noise_sigma=noise_sigma, mode=datatype, seed=SEED)

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

        #M is the number of simulations for the whole model.
        for i in range(1,M+1):
            model_data= {'N':N_train,
               'K':K,
               'y':Y[:,0],
               'X':X,
               'sigma':noise_sigma
               }

            #sm = pystan.StanModel(model_code=linear_regression_code)
            try:
                sm = pickle.load(open('model_independent_regression_13.pkl', 'rb'))
            except:
                sm = pystan.StanModel(model_code=logistic_reg_fixed_variance_code)
                with open('model_independent_regression_13.pkl', 'wb') as f:
                    pickle.dump(sm, f)

            num_proposal_samples = 6000
            #fit_hmc = sm.sampling(data=model_data, iter=800)
            fit_vb = sm.vb(data=model_data, iter=max_iters, tol_rel_obj=1e-4, output_samples=num_proposal_samples,
                           algorithm=algo, grad_samples= gradsamples, elbo_samples=elbosamples, eval_elbo=evalelbo)

            # Stan-VB
            fit_vb_samples = np.array(fit_vb['sampler_params']).T
            print(fit_vb_samples[1,:])
            stan_vb_w = fit_vb_samples[:,:K]
            stan_vb_mean = np.mean(stan_vb_w, axis=0)
            stan_vb_cov = np.cov(stan_vb_w[:,0], stan_vb_w[:,1])

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

np.save('K_hat_logistic_' + datatype + '_' + algo_name + '_' + str(N) + 'N', K_hat_stan_advi_list)

plt.figure()
plt.plot(K_list, np.nanmean(K_hat_stan_advi_list, axis=1), 'r-', alpha=1)
plt.plot(K_list, np.nanmin(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.nanmax(K_hat_stan_advi_list, axis=1), 'r-', alpha=0.5)
plt.xlabel('Dimensions')
plt.ylabel('K-hat')

np.save('K_hat_logistic_'+ datatype + '_'+ algo_name + '_' + str(N) + 'N' + '_samples_' + str(gradsamples), K_hat_stan_advi_list)
#plt.ylim((0,5))
plt.legend()
plt.savefig('Logistic_Regression_K_hat_vs_D_'  + datatype + '_' + algo_name +'_' + str(N) + 'N.pdf')