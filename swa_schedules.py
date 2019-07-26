import numpy as np

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


def stepsize_linear_adaptive_schedule(params, lr_current, lr_min, lr_max, epoch_current, epochs, swa_start, cycle_length, params_swa=None, swa_n=0):
	# assuming epoch_curent starts from index 1 and not from zero
	step_size = lr_current
	decay_rate = 0.9
	lr_multiplier = 2.5

	if params_swa is None:
		params_swa = params.copy()
	if epoch_current < swa_start:
		step_size = lr_current
	elif epoch_current >= swa_start and (epoch_current - swa_start)%cycle_length == 0:
		# this step computes the moving average which helps us to do away with storing the param values at each epoch.
		if params_swa is not None:
			means_swa = [(i*swa_n + j)/(swa_n+1) for i,j in zip(params_swa[0], params[0])]
			logsigmas_swa = [np.log((np.exp(i)*swa_n + np.exp(j))/(swa_n+1)) for i,j in zip(params_swa[1], params[1])]
		else:
			means_swa = params_swa[0]
			logsigmas_swa = params_swa[1]
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

	return step_size, params_swa, swa_n


def step_size_rms_prop_schedule(params, lr, epoch_curent, epochs, swa_start, cycle_length):
	pass




def compute_moving_avg(params1, params2, alpha):
	params1 *= (1 - alpha)*params1
	params2 += params2*alpha
	return params1, params2


def cyclical_step_size_schedule(step_size_min, step_size_max, epoch_current, cycle_length):
	t = ((epoch_current)% cycle_length +1)/ cycle_length
	step_size_current = (1. - t)*step_size_max + t*step_size_min

	return step_size_current
