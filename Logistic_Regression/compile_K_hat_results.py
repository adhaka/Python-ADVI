

import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', '-a', type=int, default=1)
parser.add_argument('--N', '-n', type=int, default=1000)
parser.add_argument('--iters', '-i', type=int, default=80000)

args = parser.parse_args()
N = args.N

if args.algorithm ==1:
    algo = 'meanfield'
    algo_name = 'mf'
elif args.algorithm ==2:
    algo = 'fullrank'
    algo_name = 'fr'


a1 = np.load('K_hat_logistic_independent_'+ algo_name + '_' + str(N) + 'N.npy')
a2 = np.load('K_hat_logistic_correlated_'+ algo_name + '_' + str(N) + 'N.npy')

K_list = [5,10,20,30,40,50,60,70,80,90,100]

num_K = a1.shape[0]
a22 = a2[:num_K,:]

K_list =  K_list[:num_K]

plt.figure()
plt.plot(K_list, np.nanmean(a1, axis=1), 'r-', alpha=1, label='independent')
plt.plot(K_list, np.nanmin(a1, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.nanmax(a1, axis=1), 'r-', alpha=0.5)
plt.plot(K_list, np.nanmean(a22, axis=1), 'b-', alpha=1, label='correlated')
plt.plot(K_list, np.nanmin(a22, axis=1), 'b-', alpha=0.5)
plt.plot(K_list, np.nanmax(a22, axis=1), 'b-', alpha=0.5)
plt.legend(loc=2)
plt.xlabel('Dimensions')
plt.ylabel('K-hat')
plt.savefig('K_hat_logistic_'+ algo_name + '_'+ str(N) +'.pdf')