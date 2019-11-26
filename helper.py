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
