import numpy as np
from autograd import grad

# Helper functions



def gaussian_entropy(log_sigma):
    return 0.5*(np.log(2*np.pi) + 2*log_sigma + 1.)

def compute_entropy(log_sigma):
    return np.sum(gaussian_entropy(log_sigma))

def reparametrize(zs, means, log_sigmas):
    samples = means[:, None] + np.exp(log_sigmas[:, None])*zs
    log_sigma_grad = (np.exp(log_sigmas[:, None])*zs).T
    return samples, log_sigma_grad


def normalise_log_iw(logw):
    iw = np.exp(logw)
    iw = iw/np.sum(iw)
    return iw, np.log(iw)


def normalise_log_iw2(logw):
    iw = np.exp(logw)


def expectation_iw(logw, statistic):
    ip_weights= np.exp(logw -np.min(logw))
    exp_statistic = ip_weights.T * statistic/ np.sum(ip_weights, axis=0)
    return exp_statistic
