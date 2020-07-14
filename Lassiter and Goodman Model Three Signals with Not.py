# Three signals with not.

import numpy
numpy.set_printoptions(linewidth = 120)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform
from scipy import integrate

###################

def theta_1_on_theta_2_distribution(theta_distribution, n):
	temp_array_0 = numpy.empty([0, n])
	for theta_2_num in range(n):
		temp_array_1 = numpy.empty(0)
		for theta_1_num in range(n):
			if theta_1_num >= theta_2_num:
				temp_value = theta_distribution.pdf(array_0[theta_1_num]) / (1. - theta_distribution.cdf(array_0[theta_2_num]))
			else:
				temp_value = 0.
			temp_array_1 = numpy.append(temp_array_1, temp_value)
		temp_array_0 = numpy.insert(temp_array_0, theta_2_num, temp_array_1, axis = 0)
	return temp_array_0

def theta_2_on_theta_1_distribution(theta_distribution, n):
	temp_array_0 = numpy.empty([n, 0])
	for theta_1_num in range(n):
		temp_array_1 = numpy.empty(0)
		for theta_2_num in range(n):
			if theta_2_num <= theta_1_num:
				temp_value = theta_distribution.pdf(array_0[theta_2_num]) / (theta_distribution.cdf(array_0[theta_1_num]))
			else:
				temp_value = 0.
			temp_array_1 = numpy.append(temp_array_1, temp_value)
		temp_array_0 = numpy.insert(temp_array_0, theta_1_num, temp_array_1, axis = 1)
	return temp_array_0

###################

def receiver_0_signal_0(h):
	return state_distribution.pdf(h)

def receiver_0_signal_1(h, theta):
	if h < theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))
		
def receiver_0_signal_2(h, theta):
	if h > theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (state_distribution.cdf(theta))

def receiver_0_not_signal_1(h, theta):
	if h >= theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (state_distribution.cdf(theta))

def receiver_0_not_signal_2(h, theta):
	if h <= theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))

def sender_1_signal_0_non_normalized(h):
	return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h)) - 0))

def sender_1_signal_1_non_normalized(h, theta):
	if h < theta:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))) - cost))

def sender_1_signal_2_non_normalized(h, theta):
	if h > theta:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (state_distribution.cdf(theta))) - cost))

def sender_1_not_signal_1_non_normalized(h, theta):
	if h >= theta:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (state_distribution.cdf(theta))) - (cost + cost_of_not)))

def sender_1_not_signal_2_non_normalized(h, theta):
	if h <= theta:
		return 0
	else:
		return numpy.exp(choice_parameter * (numpy.log(state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))) - (cost + cost_of_not)))

###################

# This is an alternative definition of receiver_0 decoding of 'not'; whereas right now we
# define 'not signal 1' as decoded the same as 'signal 2' and 'not signal 2' as decoded
# the same as 'signal 1', we could have 'not signal 1' be decoded as the renormalizing
# every value below theta_1 by the area above theta.

# def receiver_0_not_signal_1(h, theta):
# 	if h >= theta:
# 		return 0.
# 	else:
# 		return state_distribution.pdf(h) / (1 - state_distribution.cdf(theta))		
# 		
# def receiver_0_not_signal_2(h, theta):
# 	if h <= theta:
# 		return 0.
# 	else:
# 		return state_distribution.pdf(h) / (state_distribution.cdf(theta))

##########################################################################################

# Here we have the settings for a level 0 receiver decoding probabilities, given a fixed
# theta. This forms the common basis for both Lassiter and Goodman's original model and
# our modified model.

cost = 1.5
cost_of_not = 1.5/3.
choice_parameter = 4.
lower_bound = -4.
upper_bound = 4.
num_states = 80

mu = 0.
sigma = 1.
state_distribution = norm(mu,sigma)

# alpha_parameter = 1.
# beta_parameter = 9.
# location_parameter = lower_bound
# scale_parameter = upper_bound - lower_bound
# state_distribution = beta(alpha_parameter, beta_parameter, loc = location_parameter, scale = scale_parameter)

# state_distribution = uniform(lower_bound, upper_bound - lower_bound)

theta_distribution_type = 'unrelated+uniform'

if theta_distribution_type == 'normal':
	theta_distribution = norm(mu, sigma)
elif theta_distribution_type == 'Beta':
	theta_distribution = beta(3, 3, loc = lower_bound, scale = upper_bound - lower_bound)
elif theta_distribution_type == 'uniform':
	theta_distribution = uniform(lower_bound, upper_bound - lower_bound)

array_0 = numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - ((numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - numpy.linspace(lower_bound, upper_bound, num_states, endpoint = False))/2)

#########################

if theta_distribution_type == 'normal' or theta_distribution_type == 'Beta' or theta_distribution_type == 'uniform':

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution(theta_distribution, len(array_0))
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution(theta_distribution, len(array_0))

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array, axis = 1)[numpy.newaxis].T
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array, axis = 0)

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array * numpy.transpose(theta_distribution.pdf(array_0)[numpy.newaxis])
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array * theta_distribution.pdf(array_0)

	theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array)
	theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array)

	theta_2_by_theta_1_distribution_array = (theta_1_on_theta_2_distribution_array + theta_2_on_theta_1_distribution_array)/2.

	# theta_2_by_theta_1_distribution_array = theta_2_by_theta_1_distribution_array / ((upper_bound - lower_bound) / len(array_0))

	print 'theta_2_by_theta_1_distribution_array = \n%s' % theta_2_by_theta_1_distribution_array

	print numpy.sum(theta_2_by_theta_1_distribution_array)

elif theta_distribution_type == 'unrelated+uniform':
	
	theta_2_by_theta_1_distribution_array = numpy.full([num_states, num_states], 1./(num_states**2))

theta_1_distribution_array = numpy.sum(theta_2_by_theta_1_distribution_array, axis = 0)
theta_2_distribution_array = numpy.sum(theta_2_by_theta_1_distribution_array, axis = 1)

fix, ax = pyplot.subplots(1,1)
pyplot.plot(array_0, theta_1_distribution_array)
pyplot.plot(array_0, theta_2_distribution_array)
pyplot.show()

#########################

sender_1_signal_0_non_normalized_array = numpy.empty(0)
for h_num in range(len(array_0)):
	value = sender_1_signal_0_non_normalized(array_0[h_num])
	sender_1_signal_0_non_normalized_array = numpy.append(sender_1_signal_0_non_normalized_array, value)

# print 'sender_1_signal_0_non_normalized_array = \n%s' % sender_1_signal_0_non_normalized_array

sender_1_signal_1_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_signal_1_non_normalized(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_signal_1_non_normalized_array = numpy.insert(sender_1_signal_1_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_signal_1_non_normalized_array = \n%s' % sender_1_signal_1_non_normalized_array

sender_1_signal_2_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_signal_2_non_normalized(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_signal_2_non_normalized_array = numpy.insert(sender_1_signal_2_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_signal_2_non_normalized_array = \n%s' % sender_1_signal_2_non_normalized_array

sender_1_not_signal_1_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_not_signal_1_non_normalized(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_not_signal_1_non_normalized_array = numpy.insert(sender_1_not_signal_1_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_not_signal_1_non_normalized_array = \n%s' % sender_1_not_signal_1_non_normalized_array

sender_1_not_signal_2_non_normalized_array = numpy.empty([0, len(array_0)])
for theta_num in range(len(array_0)):
	temp_array = numpy.empty(0)
	for h_num in range(len(array_0)):
		value = sender_1_not_signal_2_non_normalized(array_0[h_num], array_0[theta_num])
		temp_array = numpy.append(temp_array, value)
	sender_1_not_signal_2_non_normalized_array = numpy.insert(sender_1_not_signal_2_non_normalized_array, theta_num, temp_array, axis = 0)

# print 'sender_1_not_signal_2_non_normalized_array = \n%s' % sender_1_not_signal_2_non_normalized_array

#########################

denominator_array = numpy.empty([0, len(array_0), len(array_0)])
for theta_2_num in range(len(array_0)):
	temp_array_0 = (numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) + sender_1_signal_1_non_normalized_array + numpy.tile(sender_1_signal_2_non_normalized_array[theta_2_num], [len(array_0), 1]) + sender_1_not_signal_1_non_normalized_array + numpy.tile(sender_1_not_signal_2_non_normalized_array[theta_2_num], [len(array_0), 1]))
	denominator_array = numpy.insert(denominator_array, theta_2_num, temp_array_0, axis = 0)

sender_1_signal_0_normalized_array = numpy.tile(sender_1_signal_0_non_normalized_array, (len(array_0), 1)) / denominator_array
sender_1_signal_1_normalized_array = sender_1_signal_1_non_normalized_array / denominator_array
sender_1_signal_2_normalized_array = numpy.reshape(sender_1_signal_2_non_normalized_array, (len(array_0), 1, len(array_0))) / denominator_array
sender_1_not_signal_1_normalized_array = sender_1_not_signal_1_non_normalized_array / denominator_array
sender_1_not_signal_2_normalized_array = numpy.reshape(sender_1_not_signal_2_non_normalized_array, (len(array_0), 1, len(array_0))) / denominator_array

print sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array + sender_1_not_signal_1_normalized_array + sender_1_not_signal_2_normalized_array
print numpy.sum(sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array + sender_1_not_signal_1_normalized_array + sender_1_not_signal_2_normalized_array)

sender_1_signal_0_normalized_array = sender_1_signal_0_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_signal_1_normalized_array = sender_1_signal_1_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_signal_2_normalized_array = sender_1_signal_2_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_not_signal_1_normalized_array = sender_1_not_signal_1_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))
sender_1_not_signal_2_normalized_array = sender_1_not_signal_2_normalized_array * numpy.reshape(theta_2_by_theta_1_distribution_array, (len(array_0), len(array_0), 1))

print sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array + sender_1_not_signal_1_normalized_array + sender_1_not_signal_2_normalized_array
print numpy.sum(sender_1_signal_0_normalized_array + sender_1_signal_1_normalized_array + sender_1_signal_2_normalized_array + sender_1_not_signal_1_normalized_array + sender_1_not_signal_2_normalized_array)

#########################

sender_1_signal_0_h_array = numpy.sum(numpy.sum(sender_1_signal_0_normalized_array, axis = 0), axis = 0)
sender_1_signal_1_h_array = numpy.sum(numpy.sum(sender_1_signal_1_normalized_array, axis = 0), axis = 0)
sender_1_signal_2_h_array = numpy.sum(numpy.sum(sender_1_signal_2_normalized_array, axis = 0), axis = 0)
sender_1_not_signal_1_h_array = numpy.sum(numpy.sum(sender_1_not_signal_1_normalized_array, axis = 0), axis = 0)
sender_1_not_signal_2_h_array = numpy.sum(numpy.sum(sender_1_not_signal_2_normalized_array, axis = 0), axis = 0)

#########################

fixed_theta_1_num = numpy.int(numpy.ceil(len(array_0)*(8./12.)))
fixed_theta_2_num = numpy.int(numpy.ceil(len(array_0)*(4./12.)))

print 'fixed_theta_1 = %s' % array_0[fixed_theta_1_num]
print 'fixed_theta_2 = %s' % array_0[fixed_theta_2_num]

if theta_distribution_type == 'normal' or theta_distribution_type == 'Beta' or theta_distribution_type == 'uniform':

	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_normalized_array[fixed_theta_2_num, fixed_theta_1_num]
	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array / theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_1_num]
	print 'sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array

	sender_1_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_signal_1_normalized_array[:fixed_theta_1_num + 1, fixed_theta_1_num], axis = 0)
	sender_1_signal_1_fixed_theta_1_h_array = (sender_1_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:fixed_theta_1_num + 1, fixed_theta_1_num], axis = 0))
	print 'sender_1_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_signal_1_fixed_theta_1_h_array

	sender_1_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_signal_2_normalized_array[fixed_theta_2_num, fixed_theta_2_num:], axis = 0)
	sender_1_signal_2_fixed_theta_2_h_array = (sender_1_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_2_num:]))
	print 'sender_1_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_signal_2_fixed_theta_2_h_array

	sender_1_not_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_not_signal_1_normalized_array[:fixed_theta_2_num + 1, fixed_theta_2_num], axis = 0)
	sender_1_not_signal_1_fixed_theta_1_h_array = (sender_1_not_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:fixed_theta_2_num + 1, fixed_theta_2_num], axis = 0))
	print 'sender_1_not_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_not_signal_1_fixed_theta_1_h_array

	sender_1_not_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_not_signal_2_normalized_array[fixed_theta_1_num, fixed_theta_1_num:], axis = 0)
	sender_1_not_signal_2_fixed_theta_2_h_array = (sender_1_not_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_1_num, fixed_theta_1_num:]))
	print 'sender_1_not_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_not_signal_2_fixed_theta_2_h_array

elif theta_distribution_type == 'unrelated+uniform':

	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_normalized_array[fixed_theta_2_num, fixed_theta_1_num]
	sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array / theta_2_by_theta_1_distribution_array[fixed_theta_2_num, fixed_theta_1_num]
	print 'sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array

	sender_1_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_signal_1_normalized_array[:, fixed_theta_1_num], axis = 0)
	sender_1_signal_1_fixed_theta_1_h_array = (sender_1_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:, fixed_theta_1_num], axis = 0))
	print 'sender_1_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_signal_1_fixed_theta_1_h_array

	sender_1_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_signal_2_normalized_array[fixed_theta_2_num, :], axis = 0)
	sender_1_signal_2_fixed_theta_2_h_array = (sender_1_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_2_num, :]))
	print 'sender_1_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_signal_2_fixed_theta_2_h_array

	sender_1_not_signal_1_fixed_theta_1_h_array = numpy.sum(sender_1_not_signal_1_normalized_array[:, fixed_theta_2_num], axis = 0)
	sender_1_not_signal_1_fixed_theta_1_h_array = (sender_1_not_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[:, fixed_theta_2_num], axis = 0))
	print 'sender_1_not_signal_1_fixed_theta_1_h_array = \n%s' % sender_1_not_signal_1_fixed_theta_1_h_array

	sender_1_not_signal_2_fixed_theta_2_h_array = numpy.sum(sender_1_not_signal_2_normalized_array[fixed_theta_1_num, :], axis = 0)
	sender_1_not_signal_2_fixed_theta_2_h_array = (sender_1_not_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array[fixed_theta_1_num, :]))
	print 'sender_1_not_signal_2_fixed_theta_2_h_array = \n%s' % sender_1_not_signal_2_fixed_theta_2_h_array

#########################

state_distribution_masses = state_distribution.pdf(array_0)/numpy.sum(state_distribution.pdf(array_0))

receiver_2_signal_0_array = sender_1_signal_0_normalized_array * state_distribution_masses
receiver_2_signal_1_array = sender_1_signal_1_normalized_array * state_distribution_masses
receiver_2_signal_2_array = sender_1_signal_2_normalized_array * state_distribution_masses
receiver_2_not_signal_1_array = sender_1_not_signal_1_normalized_array * state_distribution_masses
receiver_2_not_signal_2_array = sender_1_not_signal_2_normalized_array * state_distribution_masses

receiver_2_signal_0_array = receiver_2_signal_0_array / numpy.sum(receiver_2_signal_0_array)
receiver_2_signal_1_array = receiver_2_signal_1_array / numpy.sum(receiver_2_signal_1_array)
receiver_2_signal_2_array = receiver_2_signal_2_array / numpy.sum(receiver_2_signal_2_array)
receiver_2_not_signal_1_array = receiver_2_not_signal_1_array / numpy.sum(receiver_2_not_signal_1_array)
receiver_2_not_signal_2_array = receiver_2_not_signal_2_array / numpy.sum(receiver_2_not_signal_2_array)

receiver_2_signal_0_h_array = numpy.sum(numpy.sum(receiver_2_signal_0_array, axis = 0), axis = 0)
receiver_2_signal_0_h_array_densities = receiver_2_signal_0_h_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_signal_1_h_array = numpy.sum(numpy.sum(receiver_2_signal_1_array, axis = 0), axis = 0)
receiver_2_signal_1_h_array_densities = receiver_2_signal_1_h_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_signal_2_h_array = numpy.sum(numpy.sum(receiver_2_signal_2_array, axis = 0), axis = 0)
receiver_2_signal_2_h_array_densities = receiver_2_signal_2_h_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_not_signal_1_h_array = numpy.sum(numpy.sum(receiver_2_not_signal_1_array, axis = 0), axis = 0)
receiver_2_not_signal_1_h_array_densities = receiver_2_not_signal_1_h_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_not_signal_2_h_array = numpy.sum(numpy.sum(receiver_2_not_signal_2_array, axis = 0), axis = 0)
receiver_2_not_signal_2_h_array_densities = receiver_2_not_signal_2_h_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_signal_1_theta_1_array = numpy.sum(numpy.sum(receiver_2_signal_1_array, axis = 0), axis = 1)
receiver_2_signal_1_theta_1_array_densities = receiver_2_signal_1_theta_1_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_signal_2_theta_2_array = numpy.sum(numpy.sum(receiver_2_signal_2_array, axis = 1), axis = 1)
receiver_2_signal_2_theta_2_array_densities = receiver_2_signal_2_theta_2_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_not_signal_1_theta_1_array = numpy.sum(numpy.sum(receiver_2_not_signal_1_array, axis = 0), axis = 1)
receiver_2_not_signal_1_theta_1_array_densities = receiver_2_not_signal_1_theta_1_array / ((upper_bound - lower_bound)/len(array_0))

receiver_2_not_signal_2_theta_2_array = numpy.sum(numpy.sum(receiver_2_not_signal_2_array, axis = 1), axis = 1)
receiver_2_not_signal_2_theta_2_array_densities = receiver_2_not_signal_2_theta_2_array / ((upper_bound - lower_bound)/len(array_0))

#########################

fig, ax = pyplot.subplots(1, 2, figsize = (12,5))

pyplot.subplot(1, 2, 1)
line = pyplot.plot(array_0, sender_1_signal_0_h_array, lw = 2, color = 'k')
line = pyplot.plot(array_0, sender_1_signal_1_h_array, lw = 2, color = 'b')
line = pyplot.plot(array_0, sender_1_signal_2_h_array, lw = 2, color = 'r')
line = pyplot.plot(array_0, sender_1_not_signal_1_h_array, lw = 2, color = 'c')
line = pyplot.plot(array_0, sender_1_not_signal_2_h_array, lw = 2, color = 'orange')

line = pyplot.plot(array_0, sender_1_signal_0_fixed_theta_1_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'k')
line = pyplot.plot(array_0, sender_1_signal_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'b')
line = pyplot.plot(array_0, sender_1_signal_2_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'r')
line = pyplot.plot(array_0, sender_1_not_signal_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'c')
line = pyplot.plot(array_0, sender_1_not_signal_2_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'orange')

line = pyplot.plot(array_0, numpy.sum(sender_1_signal_1_normalized_array[:,:,fixed_theta_1_num], axis = 0)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 0), lw = 5, linestyle = ':', color = 'b')
line = pyplot.plot(array_0, numpy.sum(sender_1_signal_2_normalized_array[:,:,fixed_theta_2_num], axis = 1)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 1), lw = 5, linestyle = ':', color = 'r')
line = pyplot.plot(array_0, numpy.sum(sender_1_not_signal_1_normalized_array[:,:,fixed_theta_2_num], axis = 0)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 0), lw = 5, linestyle = ':', color = 'c')
line = pyplot.plot(array_0, numpy.sum(sender_1_not_signal_2_normalized_array[:,:,fixed_theta_1_num], axis = 1)/numpy.sum(theta_2_by_theta_1_distribution_array, axis = 1), lw = 5, linestyle = ':', color = 'orange')

pyplot.subplot(1, 2, 2)
line = pyplot.plot(array_0, receiver_2_signal_0_h_array_densities, lw = 2, color = 'k')
line = pyplot.plot(array_0, receiver_2_signal_1_h_array_densities, lw = 2, color = 'b')
line = pyplot.plot(array_0, receiver_2_signal_1_theta_1_array_densities, lw = 2, linestyle = '--', color = 'b')
line = pyplot.plot(array_0, receiver_2_signal_2_h_array_densities, lw = 2, color = 'r')
line = pyplot.plot(array_0, receiver_2_signal_2_theta_2_array_densities, lw = 2, linestyle = '--', color = 'r')
line = pyplot.plot(array_0, receiver_2_not_signal_1_h_array_densities, lw = 2, color = 'c')
line = pyplot.plot(array_0, receiver_2_not_signal_1_theta_1_array_densities, lw = 2, linestyle = '--', color = 'c')
line = pyplot.plot(array_0, receiver_2_not_signal_2_h_array_densities, lw = 2, color = 'orange')
line = pyplot.plot(array_0, receiver_2_not_signal_2_theta_2_array_densities, lw = 2, linestyle = '--', color = 'orange')

pyplot.subplot(1, 2, 1)
pyplot.legend([r'$\sigma_{1}(u_{0}|h)$', r'$\sigma_{1}(u_{1}|h)$', r'$\sigma_{1}(u_{2}|h)$', r'$\sigma_{1}(\neg u_{1}|h)$', r'$\sigma_{1}(\neg u_{2}|h)$', r'$\sigma_{1}(u_{0}|h, \theta_{1} \approx %s, \theta_{2} \approx %s)$' % (numpy.around(array_0[fixed_theta_1_num], decimals = 2), numpy.around(array_0[fixed_theta_2_num], decimals = 2)), r'$\sigma_{1}(u_{1}|h, \theta_{1} \approx %s)$' % numpy.around(array_0[fixed_theta_1_num], decimals = 2), r'$\sigma_{1}(u_{2}|h, \theta_{2} \approx %s)$' % numpy.around(array_0[fixed_theta_2_num], decimals = 2), r'$\sigma_{1}(\neg u_{1}|h, \theta_{1} \approx %s)$' % numpy.around(array_0[fixed_theta_2_num], decimals = 2), r'$\sigma_{1}(\neg u_{2}|h, \theta_{2} \approx %s)$' % numpy.around(array_0[fixed_theta_1_num], decimals = 2)], loc = 0, fontsize = 14)

pyplot.subplot(1, 2, 2)
pyplot.legend([r'$\rho_{2}(h|u_{0})$', r'$\rho_{2}(h|u_{1})$', r'$\rho_{2}(\theta_{1}|u_{1})$', r'$\rho_{2}(h|u_{2})$', r'$\rho_{2}(\theta_{2}|u_{2})$', r'$\rho_{2}(h|\neg u_{1})$', r'$\rho_{2}(\theta_{1}|\neg u_{1})$', r'$\rho_{2}(h|\neg u_{2})$', r'$\rho_{2}(\theta_{2}|\neg u_{2})$'], loc = 0, fontsize = 14)

fig.text(.4, 0, r'$Lassiter\ and\ Goodman\ Three\ Signals\ with\ Not$' + '\n', fontsize = 10)

fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, C(\neg u_{1}), C(\neg u_{2}) \approx %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, numpy.around(cost + cost_of_not, decimals = 2), mu, sigma, num_states, theta_distribution_type), fontsize = 10)
# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, C(\neg u_{1}), C(\neg u_{2}) \approx %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, numpy.around(cost + cost_of_not, decimals = 2), alpha_parameter, beta_parameter, num_states, theta_distribution_type), fontsize = 14)
# fig.text(.4, 0, r'$\lambda = %s, C(u_{1}), C(u_{2}) = %s, C(\neg u_{1}), C(\neg u_{2}) \approx %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s$' % (choice_parameter, cost, numpy.around(cost + cost_of_not, decimals = 2), num_states, theta_distribution_type), fontsize = 14)

# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Normal Distribution.pdf')
# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Beta Distribution.pdf')
# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Uniform Distribution.pdf')

pyplot.show()
pyplot.close()
