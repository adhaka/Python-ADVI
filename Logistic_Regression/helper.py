import numpy as np
from autograd import grad

# Helper function


def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(log_sigma):
    return np.sum(gaussian_entropy(log_sigma))

def reparametrize(zs, means, log_sigmas):
    if means.ndim ==1 :
        print('there')
        samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
        log_sigma_grad = (np.exp(log_sigmas[:, None])*zs)
        return samples, log_sigma_grad
    else:
        samples = means + np.exp(log_sigmas)*zs
        log_sigma_grad = (np.exp(log_sigmas)*zs)
    return samples, log_sigma_grad.T


def compute_distance(a, b):
    distance = np.dot(a.T, b)
    return distance

def compute_l2_norm(mu1, Sigma1, mu2, Sigma2):
    mu_dist = (mu1 - mu2) **2
    Sigma_dist = (Sigma1 - Sigma2) **2
    mu_l2 = np.sum(mu_dist)
    Sigma_l2 = np.sum(Sigma_dist)
    l2 = mu_l2 + Sigma_l2
    return l2

def compute_l2_norm_means(mu1, mu2):
    mu_dist = (mu1 - mu2) **2
    mu_l2 = np.sum(mu_dist)
    l2 = mu_l2
    return l2

def compute_l2_norm_Sigmas(Sigma1, Sigma2):
    Sigma_dist = (Sigma1 - Sigma2) **2
    Sigma_l2 = np.sum(Sigma_dist)
    l2 =  Sigma_l2
    return l2


def normalise_log_iw(logw):
    iw = np.exp(logw)
    iw = iw/np.sum(iw)
    return iw, np.log(iw)


def normalise_log_iw2(logw):
    iw = np.exp(logw)


def expectation_iw(logw, statistic):
    print(logw.shape)
    ip_weights= np.exp(logw -np.min(logw,axis=1))
    ip_weights = np.exp(logw)
    print(ip_weights.shape)
    print(statistic.shape)
    exp_statistic = ip_weights @ statistic/ (np.sum(ip_weights, axis=1) +1e-3)
    return exp_statistic


def compute_R_hat_fn(chains, warmup=500):
    #warmup = 300
    chains = chains[:, warmup:, :]
    #n_chains = N_sim
    n_iters = chains.shape[1]
    n_chains = chains.shape[0]
    K = chains.shape[2]//2
    n_iters = n_iters // 2
    psi = chains.reshape((n_chains * 2, n_iters, 2 * K))
    n_chains2 = n_chains*2
    psi_dot_j = np.mean(psi, axis=1)
    psi_dot_dot = np.mean(psi_dot_j, axis=0)
    s_j_2 = np.sum((psi - np.expand_dims(psi_dot_j, axis=1)) ** 2, axis=1) / (n_iters - 1)
    B = n_iters * np.sum((psi_dot_j - psi_dot_dot) ** 2, axis=0) / (n_chains2 - 1)
    W = np.mean(s_j_2, axis=0)
    var_hat = (n_iters - 1) * W / n_iters + (B / n_iters)
    R_hat = np.sqrt(var_hat / W)
    return R_hat
    #print(R_hat)