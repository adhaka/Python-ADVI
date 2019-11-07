

import numpy as np
import matplotlib.pyplot as plt

algo_name = 'mf'
N = 5000

a1 = np.load('K_hat_linear_independent_'+ algo_name + '_' + str(N) + 'N.npy')
a2 = np.load('K_hat_linear_correlated_'+ algo_name + '_' + str(N) + 'N.npy')

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
plt.savefig('K_hat_linear_'+ algo_name + '_'+ str(N) +'.pdf')