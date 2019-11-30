import numpy as np
from helper import compute_l2_norm
import scipy



def cyclical_step_size_schedule(step_size_min, step_size_max, epoch_current, cycle_length):
	t = ((epoch_current)% cycle_length +1)/ cycle_length
	step_size_current = (1. - t)*step_size_max + t*step_size_min
	return step_size_current


def stepsize_cyclical_adaptive_schedule(lr, epoch_current, epochs, c):
	# this is the schedule given in the blog for SWA
	step_size = 0.01
	decay_rate = 0.9
	lr_multiplier = 2.5

	if epoch_current / epochs < 0.5:
		step_size = lr
	elif epoch_current/epochs > 0.5 and epoch_current/epochs < 0.75:
		step_size = lr * decay_rate
	else:
		step_size = cyclical_step_size_schedule(lr, 2.5*lr, epoch_current, c)
	return step_size


def stepsize_linear_adaptive_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0, weight=1.):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	weight= 1.0

	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.
		if params_swa is not None:
			means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
		else:
			means_swa = params[0]
			logsigmas_swa = params[1]
		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		step_size = lr_max
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n


def stepsize_linear_adaptive_is_mixing_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, densities, params_swa=None, swa_n=0, weight=1.):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	weight= 4.5
	debug = False
	new_densities = dict()
	new_densities['sd'] =densities['sd']
	new_densities['sum'] = densities['sum']
	new_densities = densities.copy()

	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.
		sample_density = densities['sd']
		sample_density_sum = densities['sum']

		if params_swa is not None:
			print('swa computations')
			print(sample_density_sum)
			print(sample_density)

			means_swa = [(i*sample_density_sum + j*sample_density)/(sample_density_sum + sample_density) for i,j in zip(params_swa[0], params[0])]
			#logsigmas_swa = [np.log((np.exp(i)*sample_density_sum + np.exp(j)*sample_density)/(sample_density_sum + sample_density)) for i,j in zip(params_swa[1], params[1])]
			logsigmas_swa = [(i*sample_density_sum + j*sample_density)/(sample_density_sum + sample_density) for i,j in zip(params_swa[1], params[1])]


			if debug:
				print(params_swa[0])
				print(params[0])
				print(params[1])
				print(params_swa[1])
				print(np.array(means_swa))
				print(np.array(logsigmas_swa))
				print(sample_density_sum)
				print(sample_density)

#			if swa_n == 1:
#				exit()


			#means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			#logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
			sample_density_sum += sample_density
			#mean_density_sum += mean_density
			new_densities['sum'] = sample_density_sum
		else:
			means_swa = params[0]
			logsigmas_swa = params[1]
			sample_density_sum = sample_density

		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		# print(params)
		# print(params_swa)
		# print(epoch_current)
		# print(swa_start)
		# increase the current reduced learning rate in order to explore more ...
		#step_size = step_size*lr_multiplier
		step_size = lr_max
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

	return step_size, params_swa, swa_n, new_densities

def stepsize_adaptive_swa_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5
	weight = 3
	if params_swa is None:
		params_swa = params.copy()

	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		step_size = lr_max
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 1:
		if params_swa is not None:
			means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
			logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
		else:
			means_swa = params_swa[0]
			logsigmas_swa = params_swa[1]
		swa_n += 1
		n_models = 1 + (epoch_current - swa_start) / cycle_length
		params_swa = [means_swa, logsigmas_swa]
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)
	else:
		# print(step_size)
		step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)
	return step_size, params_swa, swa_n


def stepsize_linear_weight_averaging(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0, weight=1, pmz='std'):
    step_size = lr_current
    decay_rate= 0.9
    lr_multiplier= 2.5
    if params_swa is None:
        params_swa = params.copy()

    if epoch_current < swa_start:
        means_swa = params[0]
        logsigmas_swa = params[1]
        params_swa = [means_swa, logsigmas_swa]
        step_size = lr_current
    elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
        if pmz == 'log':
            means_swa = [(i*swa_n*weight + j)/(swa_n*weight+1) for i,j in zip(params_swa[0], params[0])]
            logsigmas_swa = [np.log((np.exp(i)*swa_n*weight + np.exp(j))/(swa_n*weight+1)) for i,j in zip(params_swa[1], params[1])]
        else:
            means_swa = [(i * swa_n * weight + j) / (swa_n * weight + 1) for i, j in zip(params_swa[0], params[0])]
            logsigmas_swa = [(i * swa_n * weight + j) / (swa_n * weight + 1) for i, j in zip(params_swa[1], params[1])]

        swa_n +=1
        n_models = 1 + (epoch_current - swa_start) / cycle_length
        params_swa = [means_swa, logsigmas_swa]
        step_size = lr_max
    else:
        step_size = cyclical_step_size_schedule(lr_min, lr_max, epoch_current, cycle_length)

    return step_size, params_swa, swa_n


def rms_prop_gradient(epoch_current, mean_grads, sigma_grads, s_previous=None, eta=0.1):
	grad_vec= np.concatenate((mean_grads, sigma_grads), axis=0)
	if epoch_current <= 1:
		s = 0.5*grad_vec**2
	else:
		s= 0.2*grad_vec**2 + 0.7*s_previous

	rho = epoch_current**(-0.5 + 1e-12)*eta/(1. +np.sqrt(s))
	return rho, s


def step_size_rms_prop_schedule(params_rms, epoch_current, mean_grads, sigma_grads, s_previous=None, eta=0.1):
	grad_vec= np.concatenate((mean_grads.flatten(), sigma_grads.flatten()), axis=0)

	#print(grad_vec)

	K= mean_grads.size
	if epoch_current <= 1:
		s = 0.2*grad_vec**2
	else:
		s= 0.2*grad_vec**2 + (0.8)*s_previous

	means, betas  = params_rms[0], params_rms[1]
	rho = epoch_current**(-0.2 + 1e-16)*eta /(1. +np.sqrt(s))
	means = means + rho[:K]*mean_grads
	betas = betas + rho[K:]*sigma_grads


	# if epoch_current % 10000 == 0:
	# 	print(epoch_current)
	# 	print(means)
	# 	print(betas)
	# 	print('sigma diff:')
	# 	print(rho[K:] *sigma_grads)
		#exit()
	#params_new = np.concatenate((means.flatten(), betas.flatten()), axis=0)
	params_new = [means, betas]
	return rho, s, params_new


def compute_moving_avg(params1, params2, alpha):
	params1 *= (1 - alpha)*params1
	params2 += params2*alpha
	return params1, params2


logit = lambda x: 1./ (1 +np.exp(-x))

def cyclical_step_size_schedule(step_size_min, step_size_max, epoch_current, cycle_length):
	t = ((epoch_current)% cycle_length +1)/ cycle_length
	step_size_current = (1. - t)*step_size_max + t*step_size_min

	return step_size_current
