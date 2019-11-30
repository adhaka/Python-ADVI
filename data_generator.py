

import numpy as np
import scipy.stats

logit = lambda x: 1./ (1 +np.exp(-x))

def data_generator_linear(N, K, noise_sigma=1., mode='independent', seed=100):
    np.random.seed(seed=seed)
    mean_val = 5
    noise_var = noise_sigma ** 2
    M=1
    alpha = 1.01
    alpha_sqrt = np.sqrt(alpha)

    if mode == 'independent':
        Posterior_Sigma = np.eye(K)
    elif mode == 'correlated':
        correlation_factor=0.01
        Posterior_Sigma = np.eye(K) + (np.ones((K, K))  - np.eye(K)) * correlation_factor

    Posterior_Sigma_inv = np.linalg.inv(Posterior_Sigma)
    X_var = (Posterior_Sigma_inv - np.eye(K) / alpha) * noise_var
    X_mean = np.zeros(K)
    X = np.random.multivariate_normal(X_mean, X_var, N)
    W= np.random.normal(mean_val, alpha_sqrt, (K,1))
    y_mean = X @ W
    Y  = y_mean + np.random.normal(0, noise_sigma, (N, M))
    regression_data={}
    regression_data['X'] = X
    regression_data['Y'] = Y
    regression_data['W'] = W
    return regression_data


def data_generator_logistic(N, K, noise_sigma, mode='independent', seed=100):
    regression_data = data_generator_linear(N, K, noise_sigma, mode, seed=seed)
    Y_linear = regression_data['Y']
    p_full = logit(Y_linear)
    y_full = np.random.binomial(n=1, p=p_full)
    regression_data['Y'] = y_full
    return regression_data
