# Here we allow arbitrary iteration up the hierarchy.

import numpy
numpy.set_printoptions(linewidth = 120)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import beta
from scipy.stats import uniform
from scipy.stats import truncnorm
from scipy import integrate

###################

def initial_theta_1_on_theta_2_distribution(theta_distribution, n):
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

def initial_theta_2_on_theta_1_distribution(theta_distribution, n):
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

def theta_1_on_theta_2_distribution(theta_distribution, n):
	temp_array_0 = numpy.empty([0, n])
	for theta_2_num in range(n):
		temp_array_1 = numpy.empty(0)
		for theta_1_num in range(n):
			if theta_1_num >= theta_2_num:
				temp_value = theta_distribution[theta_1_num] / numpy.sum(theta_distribution[theta_2_num:])
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
				temp_value = theta_distribution[theta_2_num] / numpy.sum(theta_distribution[:theta_1_num + 1])
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

def receiver_0_signal_not_1(h, theta):
	if h > theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (state_distribution.cdf(theta))

def receiver_0_signal_not_2(h, theta):
	if h < theta:
		return 0.
	else:
		return state_distribution.pdf(h) / (1. - state_distribution.cdf(theta))

def pragmatic_sender(n):
	if n not in pragmatic_sender_memo:
		print 'pragmatic_receiver(%s - 1)[0] = %s' % (n, pragmatic_receiver(n - 1)[0])
		pragmatic_receiver_signal_0_h_array = numpy.sum(pragmatic_receiver(n - 1)[0], axis = (0, 1))
		print 'level %s pragmatic_receiver_signal_0_h_array = %s' % (n, pragmatic_receiver_signal_0_h_array)
		pragmatic_receiver_signal_1_theta_1_by_h_array = numpy.sum(pragmatic_receiver(n - 1)[1], axis = 0)
		pragmatic_receiver_signal_2_theta_2_by_h_array = numpy.sum(pragmatic_receiver(n - 1)[2], axis = 1)
		pragmatic_receiver_signal_not_1_theta_1_by_h_array = numpy.sum(pragmatic_receiver(n - 1)[3], axis = 0)
		pragmatic_receiver_signal_not_2_theta_2_by_h_array = numpy.sum(pragmatic_receiver(n - 1)[4], axis = 1)
		
# note that we have two options here: first we could make the pragmatic sender sensitive
# to just h, second we could make it sensitive to h and theta. of course in Lassiter and
# Goodman's model the first level sender is only sensitive to h. if we want to keep a
# single model all the way up the hierarchy it seems like making the sender only h
# sensitive is the way to go. there is also a question about theta values: do we get them
# from the same level or the previous level? What way to think about it is consistent with
# the initial model? does the initial model (the first level sender) prescribe only one
# way of doing the calculations? Perhaps one way is to think of the receiver at any level
# as taking the background knowledge of h and then applying a literal interpretation which
# simply renormalizes above any given theta. at the bottom level 0 the background
# information is an absolute prior, and at level 2 the background for any given signal is
# the h posterior. Then the level 3 sender thinks of the level 2 receiver as renormalizing
# for any given theta above (or below) the h posterior for the given signal. That can give
# rise to a level 4 receiver who in turn derives a new posterior. Is this any different
# than what we have below for h sensitive?
		
		if pragmatic_sender_type == 'h_sensitive_version_3':
		
			pragmatic_receiver_signal_0_h_array = pragmatic_receiver_signal_0_h_array
			pragmatic_receiver_signal_1_h_array = numpy.sum(pragmatic_receiver(n - 1)[1], axis = (0, 1))
			pragmatic_receiver_signal_2_h_array = numpy.sum(pragmatic_receiver(n - 1)[2], axis = (0, 1))
			pragmatic_receiver_signal_not_1_h_array = numpy.sum(pragmatic_receiver(n - 1)[3], axis = (0, 1))
			pragmatic_receiver_signal_not_2_h_array = numpy.sum(pragmatic_receiver(n - 1)[4], axis = (0, 1))
			
			print 'pragmatic_receiver_signal_0_h_array = \n%s' % pragmatic_receiver_signal_0_h_array
			print 'pragmatic_receiver_signal_1_h_array = \n%s' % pragmatic_receiver_signal_1_h_array
			print 'pragmatic_receiver_signal_2_h_array = \n%s' % pragmatic_receiver_signal_2_h_array
			print 'pragmatic_receiver_signal_not_1_h_array = \n%s' % pragmatic_receiver_signal_not_1_h_array
			print 'pragmatic_receiver_signal_not_2_h_array = \n%s' % pragmatic_receiver_signal_not_2_h_array
			
			
			pragmatic_sender_signal_0_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.tile(pragmatic_receiver_signal_0_h_array / (num_states ** 2), (num_states, num_states, 1))) - cost_of_null_signal))
			pragmatic_sender_signal_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.tile(pragmatic_receiver_signal_1_h_array / (num_states ** 2), (num_states, num_states, 1))) - cost))
			pragmatic_sender_signal_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.tile(pragmatic_receiver_signal_2_h_array / (num_states ** 2), (num_states, num_states, 1))) - cost))
			pragmatic_sender_signal_not_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.tile(pragmatic_receiver_signal_not_1_h_array / (num_states ** 2), (num_states, num_states, 1))) - (cost + cost_of_not)))
			pragmatic_sender_signal_not_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.tile(pragmatic_receiver_signal_not_2_h_array / (num_states ** 2), (num_states, num_states, 1))) - (cost + cost_of_not)))
			
			print 'pragmatic_sender_signal_0_non_normalized = \n%s' % pragmatic_sender_signal_0_non_normalized
			print 'pragmatic_sender_signal_1_non_normalized = \n%s' % pragmatic_sender_signal_1_non_normalized
			print 'pragmatic_sender_signal_2_non_normalized = \n%s' % pragmatic_sender_signal_2_non_normalized
			print 'pragmatic_sender_signal_not_1_non_normalized = \n%s' % pragmatic_sender_signal_not_1_non_normalized
			print 'pragmatic_sender_signal_not_2_non_normalized = \n%s' % pragmatic_sender_signal_not_2_non_normalized
			
			denominator_array = pragmatic_sender_signal_0_non_normalized + pragmatic_sender_signal_1_non_normalized + pragmatic_sender_signal_2_non_normalized + pragmatic_sender_signal_not_1_non_normalized + pragmatic_sender_signal_not_2_non_normalized
		
			pragmatic_sender_signal_0_normalized = pragmatic_sender_signal_0_non_normalized / denominator_array
			pragmatic_sender_signal_1_normalized = pragmatic_sender_signal_1_non_normalized / denominator_array
			pragmatic_sender_signal_2_normalized = pragmatic_sender_signal_2_non_normalized / denominator_array
			pragmatic_sender_signal_not_1_normalized = pragmatic_sender_signal_not_1_non_normalized / denominator_array
			pragmatic_sender_signal_not_2_normalized = pragmatic_sender_signal_not_2_non_normalized / denominator_array
		
			print 'level %s pragmatic_sender_signal_0_normalized = \n%s' % (n, pragmatic_sender_signal_0_normalized)
			print 'level %s pragmatic_sender_signal_1_normalized = \n%s' % (n, pragmatic_sender_signal_1_normalized)
			print 'level %s pragmatic_sender_signal_2_normalized = \n%s' % (n, pragmatic_sender_signal_2_normalized)
			print 'level %s pragmatic_sender_signal_not_1_normalized = \n%s' % (n, pragmatic_sender_signal_not_1_normalized)
			print 'level %s pragmatic_sender_signal_not_2_normalized = \n%s' % (n, pragmatic_sender_signal_not_2_normalized)

		elif pragmatic_sender_type == 'h_sensitive':
		
			pragmatic_receiver_signal_0_pre_normalization_factor_array = numpy.ones(num_states)			
			pragmatic_receiver_signal_1_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[1], axis = (0, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_2_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[2], axis = (1, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_not_1_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[3], axis = (0, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_not_2_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[4], axis = (1, 2)), (num_states, 1)).T
			
			print 'pragmatic_receiver_signal_0_h_array = \n%s' % pragmatic_receiver_signal_0_h_array
			print 'pragmatic_receiver_signal_1_theta_1_by_h_array = \n%s' % pragmatic_receiver_signal_1_theta_1_by_h_array
			print 'pragmatic_receiver_signal_2_theta_2_by_h_array = \n%s' % pragmatic_receiver_signal_2_theta_2_by_h_array
			print 'pragmatic_receiver_signal_not_1_theta_1_by_h_array = \n%s' % pragmatic_receiver_signal_not_1_theta_1_by_h_array
			print 'pragmatic_receiver_signal_not_2_theta_2_by_h_array = \n%s' % pragmatic_receiver_signal_not_2_theta_2_by_h_array
						
			print 'pragmatic_receiver_signal_0_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_0_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_1_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_1_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_2_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_2_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_not_1_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_not_1_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_not_2_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_not_2_pre_normalization_factor_array
						
			pragmatic_sender_signal_0_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_0_h_array / pragmatic_receiver_signal_0_pre_normalization_factor_array) - cost_of_null_signal))
			pragmatic_sender_signal_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_1_theta_1_by_h_array / pragmatic_receiver_signal_1_pre_normalization_factor_array) - cost))
			pragmatic_sender_signal_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_2_theta_2_by_h_array / pragmatic_receiver_signal_2_pre_normalization_factor_array) - cost))
			pragmatic_sender_signal_not_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_not_1_theta_1_by_h_array / pragmatic_receiver_signal_not_1_pre_normalization_factor_array) - (cost + cost_of_not)))
			pragmatic_sender_signal_not_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_not_2_theta_2_by_h_array / pragmatic_receiver_signal_not_2_pre_normalization_factor_array) - (cost + cost_of_not)))

			print 'pragmatic_sender_signal_0_non_normalized = \n%s' % pragmatic_sender_signal_0_non_normalized
			print 'pragmatic_sender_signal_1_non_normalized = \n%s' % pragmatic_sender_signal_1_non_normalized
			print 'pragmatic_sender_signal_2_non_normalized = \n%s' % pragmatic_sender_signal_2_non_normalized
			print 'pragmatic_sender_signal_not_1_non_normalized = \n%s' % pragmatic_sender_signal_not_1_non_normalized
			print 'pragmatic_sender_signal_not_2_non_normalized = \n%s' % pragmatic_sender_signal_not_2_non_normalized
		
			denominator_array = numpy.empty([0, num_states, num_states])
			for theta_2_num in range(num_states):
				temp_array_0 = (numpy.tile(pragmatic_sender_signal_0_non_normalized, (num_states, 1)) + pragmatic_sender_signal_1_non_normalized + numpy.tile(pragmatic_sender_signal_2_non_normalized[theta_2_num], [num_states, 1]) + pragmatic_sender_signal_not_1_non_normalized + numpy.tile(pragmatic_sender_signal_not_2_non_normalized[theta_2_num], [num_states, 1]))
				denominator_array = numpy.insert(denominator_array, theta_2_num, temp_array_0, axis = 0)
		
			pragmatic_sender_signal_0_normalized = numpy.tile(pragmatic_sender_signal_0_non_normalized, (num_states, 1)) / denominator_array
			pragmatic_sender_signal_1_normalized = pragmatic_sender_signal_1_non_normalized / denominator_array
			pragmatic_sender_signal_2_normalized = numpy.reshape(pragmatic_sender_signal_2_non_normalized, (num_states, 1, num_states)) / denominator_array
			pragmatic_sender_signal_not_1_normalized = pragmatic_sender_signal_not_1_non_normalized / denominator_array
			pragmatic_sender_signal_not_2_normalized = numpy.reshape(pragmatic_sender_signal_not_2_non_normalized, (num_states, 1, num_states)) / denominator_array
		
			print 'level %s pragmatic_sender_signal_0_normalized = \n%s' % (n, pragmatic_sender_signal_0_normalized)
			print 'level %s pragmatic_sender_signal_1_normalized = \n%s' % (n, pragmatic_sender_signal_1_normalized)
			print 'level %s pragmatic_sender_signal_2_normalized = \n%s' % (n, pragmatic_sender_signal_2_normalized)
			print 'level %s pragmatic_sender_signal_not_1_normalized = \n%s' % (n, pragmatic_sender_signal_not_1_normalized)
			print 'level %s pragmatic_sender_signal_not_2_normalized = \n%s' % (n, pragmatic_sender_signal_not_2_normalized)

		elif pragmatic_sender_type == 'h_and_theta_sensitive':
		
			pragmatic_receiver_signal_0_pre_normalization_factor_array = numpy.ones(num_states)	* num_states		
			pragmatic_receiver_signal_1_pre_normalization_factor_array = numpy.ones((num_states, num_states))
			pragmatic_receiver_signal_2_pre_normalization_factor_array = numpy.ones((num_states, num_states))
			pragmatic_receiver_signal_not_1_pre_normalization_factor_array = numpy.ones((num_states, num_states))
			pragmatic_receiver_signal_not_2_pre_normalization_factor_array = numpy.ones((num_states, num_states))
			
			pragmatic_sender_signal_0_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_0_h_array / pragmatic_receiver_signal_0_pre_normalization_factor_array) - cost_of_null_signal))
			pragmatic_sender_signal_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_1_theta_1_by_h_array / pragmatic_receiver_signal_1_pre_normalization_factor_array) - cost))
			pragmatic_sender_signal_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_2_theta_2_by_h_array / pragmatic_receiver_signal_2_pre_normalization_factor_array) - cost))
			pragmatic_sender_signal_not_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_not_1_theta_1_by_h_array / pragmatic_receiver_signal_not_1_pre_normalization_factor_array) - (cost + cost_of_not)))
			pragmatic_sender_signal_not_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(pragmatic_receiver_signal_not_2_theta_2_by_h_array / pragmatic_receiver_signal_not_2_pre_normalization_factor_array) - (cost + cost_of_not)))

			print 'pragmatic_sender_signal_0_non_normalized = \n%s' % pragmatic_sender_signal_0_non_normalized
			print 'pragmatic_sender_signal_1_non_normalized = \n%s' % pragmatic_sender_signal_1_non_normalized
			print 'pragmatic_sender_signal_2_non_normalized = \n%s' % pragmatic_sender_signal_2_non_normalized
			print 'pragmatic_sender_signal_not_1_non_normalized = \n%s' % pragmatic_sender_signal_not_1_non_normalized
			print 'pragmatic_sender_signal_not_2_non_normalized = \n%s' % pragmatic_sender_signal_not_2_non_normalized
		
			denominator_array = numpy.empty([0, num_states, num_states])
			for theta_2_num in range(num_states):
				temp_array_0 = (numpy.tile(pragmatic_sender_signal_0_non_normalized, (num_states, 1)) + pragmatic_sender_signal_1_non_normalized + numpy.tile(pragmatic_sender_signal_2_non_normalized[theta_2_num], [num_states, 1]) + pragmatic_sender_signal_not_1_non_normalized + numpy.tile(pragmatic_sender_signal_not_2_non_normalized[theta_2_num], [num_states, 1]))
				denominator_array = numpy.insert(denominator_array, theta_2_num, temp_array_0, axis = 0)
		
			pragmatic_sender_signal_0_normalized = numpy.tile(pragmatic_sender_signal_0_non_normalized, (num_states, 1)) / denominator_array
			pragmatic_sender_signal_1_normalized = pragmatic_sender_signal_1_non_normalized / denominator_array
			pragmatic_sender_signal_2_normalized = numpy.reshape(pragmatic_sender_signal_2_non_normalized, (num_states, 1, num_states)) / denominator_array
			pragmatic_sender_signal_not_1_normalized = pragmatic_sender_signal_not_1_non_normalized / denominator_array
			pragmatic_sender_signal_not_2_normalized = numpy.reshape(pragmatic_sender_signal_not_2_non_normalized, (num_states, 1, num_states)) / denominator_array
		
			print 'level %s pragmatic_sender_signal_0_normalized = \n%s' % (n, pragmatic_sender_signal_0_normalized)
			print 'level %s pragmatic_sender_signal_1_normalized = \n%s' % (n, pragmatic_sender_signal_1_normalized)
			print 'level %s pragmatic_sender_signal_2_normalized = \n%s' % (n, pragmatic_sender_signal_2_normalized)
			print 'level %s pragmatic_sender_signal_not_1_normalized = \n%s' % (n, pragmatic_sender_signal_not_1_normalized)
			print 'level %s pragmatic_sender_signal_not_2_normalized = \n%s' % (n, pragmatic_sender_signal_not_2_normalized)

		elif pragmatic_sender_type == 'modified h sensitive':
		
			pragmatic_receiver_signal_0_pre_normalization_factor_array = numpy.ones((num_states, num_states))			
			pragmatic_receiver_signal_1_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[1], axis = (0, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_2_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[2], axis = (1, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_not_1_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[3], axis = (0, 2)), (num_states, 1)).T
			pragmatic_receiver_signal_not_2_pre_normalization_factor_array = numpy.tile(numpy.sum(pragmatic_receiver(n - 1)[4], axis = (1, 2)), (num_states, 1)).T
			
			print 'pragmatic_receiver_signal_0_h_array = \n%s' % pragmatic_receiver_signal_0_h_array
			print 'pragmatic_receiver_signal_1_theta_1_by_h_array = \n%s' % pragmatic_receiver_signal_1_theta_1_by_h_array
			print 'pragmatic_receiver_signal_2_theta_2_by_h_array = \n%s' % pragmatic_receiver_signal_2_theta_2_by_h_array
			print 'pragmatic_receiver_signal_not_1_theta_1_by_h_array = \n%s' % pragmatic_receiver_signal_not_1_theta_1_by_h_array
			print 'pragmatic_receiver_signal_not_2_theta_2_by_h_array = \n%s' % pragmatic_receiver_signal_not_2_theta_2_by_h_array
						
			print 'pragmatic_receiver_signal_0_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_0_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_1_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_1_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_2_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_2_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_not_1_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_not_1_pre_normalization_factor_array
			print 'pragmatic_receiver_signal_not_2_pre_normalization_factor_array = \n%s' % pragmatic_receiver_signal_not_2_pre_normalization_factor_array
						
			pragmatic_sender_signal_0_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.sum(weighting_array.reshape(1, num_states, num_states) * (pragmatic_receiver_signal_0_h_array / pragmatic_receiver_signal_0_pre_normalization_factor_array).reshape(num_states, 1, num_states), axis = 2)) - cost_of_null_signal))
			pragmatic_sender_signal_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.sum(weighting_array.reshape(1, num_states, num_states) * (pragmatic_receiver_signal_1_theta_1_by_h_array / pragmatic_receiver_signal_1_pre_normalization_factor_array).reshape(num_states, 1, num_states), axis = 2)) - cost))
			pragmatic_sender_signal_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.sum(weighting_array.reshape(1, num_states, num_states) * (pragmatic_receiver_signal_2_theta_2_by_h_array / pragmatic_receiver_signal_2_pre_normalization_factor_array).reshape(num_states, 1, num_states), axis = 2)) - cost))
			pragmatic_sender_signal_not_1_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.sum(weighting_array.reshape(1, num_states, num_states) * (pragmatic_receiver_signal_not_1_theta_1_by_h_array / pragmatic_receiver_signal_not_1_pre_normalization_factor_array).reshape(num_states, 1, num_states), axis = 2)) - (cost + cost_of_not)))
			pragmatic_sender_signal_not_2_non_normalized = numpy.exp(choice_parameter * (numpy.log(numpy.sum(weighting_array.reshape(1, num_states, num_states) * (pragmatic_receiver_signal_not_2_theta_2_by_h_array / pragmatic_receiver_signal_not_2_pre_normalization_factor_array).reshape(num_states, 1, num_states), axis = 2)) - (cost + cost_of_not)))

			print 'pragmatic_sender_signal_0_non_normalized = \n%s' % pragmatic_sender_signal_0_non_normalized
			print 'pragmatic_sender_signal_1_non_normalized = \n%s' % pragmatic_sender_signal_1_non_normalized
			print 'pragmatic_sender_signal_2_non_normalized = \n%s' % pragmatic_sender_signal_2_non_normalized
			print 'pragmatic_sender_signal_not_1_non_normalized = \n%s' % pragmatic_sender_signal_not_1_non_normalized
			print 'pragmatic_sender_signal_not_2_non_normalized = \n%s' % pragmatic_sender_signal_not_2_non_normalized
		
			denominator_array = numpy.empty([0, num_states, num_states])
			for theta_2_num in range(num_states):
				temp_array_0 = (pragmatic_sender_signal_0_non_normalized + pragmatic_sender_signal_1_non_normalized + numpy.tile(pragmatic_sender_signal_2_non_normalized[theta_2_num], [num_states, 1]) + pragmatic_sender_signal_not_1_non_normalized + numpy.tile(pragmatic_sender_signal_not_2_non_normalized[theta_2_num], [num_states, 1]))
				denominator_array = numpy.insert(denominator_array, theta_2_num, temp_array_0, axis = 0)
		
			pragmatic_sender_signal_0_normalized = pragmatic_sender_signal_0_non_normalized / denominator_array
			pragmatic_sender_signal_1_normalized = pragmatic_sender_signal_1_non_normalized / denominator_array
			pragmatic_sender_signal_2_normalized = numpy.reshape(pragmatic_sender_signal_2_non_normalized, (num_states, 1, num_states)) / denominator_array
			pragmatic_sender_signal_not_1_normalized = pragmatic_sender_signal_not_1_non_normalized / denominator_array
			pragmatic_sender_signal_not_2_normalized = numpy.reshape(pragmatic_sender_signal_not_2_non_normalized, (num_states, 1, num_states)) / denominator_array
		
			print 'level %s pragmatic_sender_signal_0_normalized = \n%s' % (n, pragmatic_sender_signal_0_normalized)
			print 'level %s pragmatic_sender_signal_1_normalized = \n%s' % (n, pragmatic_sender_signal_1_normalized)
			print 'level %s pragmatic_sender_signal_2_normalized = \n%s' % (n, pragmatic_sender_signal_2_normalized)
			print 'level %s pragmatic_sender_signal_not_1_normalized = \n%s' % (n, pragmatic_sender_signal_not_1_normalized)
			print 'level %s pragmatic_sender_signal_not_2_normalized = \n%s' % (n, pragmatic_sender_signal_not_2_normalized)
		
		pragmatic_sender_memo[n] = numpy.asarray((pragmatic_sender_signal_0_normalized, pragmatic_sender_signal_1_normalized, pragmatic_sender_signal_2_normalized, pragmatic_sender_signal_not_1_normalized, pragmatic_sender_signal_not_2_normalized))
	return pragmatic_sender_memo[n]

def pragmatic_receiver(n):
	if n not in pragmatic_receiver_memo:

		if theta_posterior_source == 'signal 1, signal 2':
			theta_1_distribution_array = numpy.sum(pragmatic_receiver(n-2)[1:2], axis = (0, 1, 3))
			theta_2_distribution_array = numpy.sum(pragmatic_receiver(n-2)[2:3], axis = (0, 2, 3))		
		elif theta_posterior_source == 'signal 1, not 1, signal 2, not 2':
			theta_1_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[1:2], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[3:4], axis = (0, 1, 3))) / 2.
			theta_2_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[2:3], axis = (0, 2, 3)) + numpy.sum(pragmatic_receiver(n-2)[4:5], axis = (0, 2, 3))) / 2.		
		elif theta_posterior_source == 'signal 1, 2, not 1, not 2, signal 1, 2, not 1, not 2':
			theta_1_distribution_array = numpy.sum(pragmatic_receiver(n-2)[1:], axis = (0, 1, 3)) / 4.
			theta_2_distribution_array = numpy.sum(pragmatic_receiver(n-2)[1:], axis = (0, 2, 3)) / 4.
		elif theta_posterior_source == 'signal 0, 1, signal 0, 2':
			theta_1_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[0:1], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[1:2], axis = (0, 1, 3))) / 2.
			theta_2_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[0:1], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[2:3], axis = (0, 2, 3))) / 2.	
		elif theta_posterior_source == 'signal 0, 1, not 1, signal 0, 2, not 2':
			theta_1_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[0:1], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[1:2], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[3:4], axis = (0, 1, 3))) / 3.
			theta_2_distribution_array = (numpy.sum(pragmatic_receiver(n-2)[0:1], axis = (0, 1, 3)) + numpy.sum(pragmatic_receiver(n-2)[2:3], axis = (0, 2, 3)) + numpy.sum(pragmatic_receiver(n-2)[4:5], axis = (0, 2, 3))) / 3.		
		elif theta_posterior_source == 'signal 0, 1, 2, not 1, not 2, signal 0, 1, 2, not 1, not 2':
			theta_1_distribution_array = numpy.sum(pragmatic_receiver(n-2)[0:], axis = (0, 1, 3)) / 5.
			theta_2_distribution_array = numpy.sum(pragmatic_receiver(n-2)[0:], axis = (0, 2, 3)) / 5.
		elif theta_posterior_source == 'signal_specific':
			theta_2_by_theta_1_distribution_array_array = numpy.sum(pragmatic_receiver(n-2), axis = 3)

		if theta_distribution_relation == 'True':
			if theta_posterior_source != 'signal_specific':
				theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution(theta_1_distribution_array, num_states)
				theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array * numpy.transpose(theta_2_distribution_array[numpy.newaxis])
				theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array)

				theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution(theta_2_distribution_array, num_states)
				theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array * theta_1_distribution_array
				theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array)

				theta_2_by_theta_1_distribution_array = (theta_1_on_theta_2_distribution_array + theta_2_on_theta_1_distribution_array)/2.

			else:
				temp_array = numpy.empty([0, num_states, num_states])
				for array_num in range(len(theta_2_by_theta_1_distribution_array_array)):
					theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution(numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 0), num_states)
					theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array * numpy.transpose(numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 1)[numpy.newaxis])
					theta_1_on_theta_2_distribution_array = theta_1_on_theta_2_distribution_array / numpy.sum(theta_1_on_theta_2_distribution_array)

					theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution(numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 1), num_states)
					theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array * numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 0)
					theta_2_on_theta_1_distribution_array = theta_2_on_theta_1_distribution_array / numpy.sum(theta_2_on_theta_1_distribution_array)

					theta_2_by_theta_1_distribution_array = (theta_1_on_theta_2_distribution_array + theta_2_on_theta_1_distribution_array)/2.

					temp_array = numpy.insert(temp_array, array_num, theta_2_by_theta_1_distribution_array, axis = 0)
				theta_2_by_theta_1_distribution_array_array = temp_array

		elif theta_distribution_relation == 'False':		
			if theta_posterior_source != 'signal_specific':
				theta_2_by_theta_1_distribution_array = theta_1_distribution_array * theta_2_distribution_array[numpy.newaxis].T
		
		if theta_posterior_source != 'signal_specific':
			theta_2_by_theta_1_distribution_array_array = numpy.tile(theta_2_by_theta_1_distribution_array, (5, 1, 1))

		for array_num in range(len(theta_2_by_theta_1_distribution_array_array)):

			plot_theta_1_distribution_array = numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 0)
			plot_theta_2_distribution_array = numpy.sum(theta_2_by_theta_1_distribution_array_array[array_num], axis = 1)

			fig, ax = pyplot.subplots(1,1)
			pyplot.plot(array_0, plot_theta_1_distribution_array)
			pyplot.plot(array_0, plot_theta_2_distribution_array)
			pyplot.legend([r'$\theta_{1}$', r'$\theta_{2}$'], loc = 0, fontsize = 28)
			pyplot.show()

			fig = pyplot.figure()
			ax = fig.gca(projection = '3d')
			ax.set_xlim(-4., 4.)
			ax.set_ylim(-4., 4.)
			surface = ax.plot_surface(numpy.tile(array_0, (num_states, 1)), numpy.tile(array_0, (num_states, 1)).T, theta_2_by_theta_1_distribution_array_array[array_num], cmap = cm.coolwarm, linewidth = 0.0, antialiased = True, rstride = 2, cstride = 2, shade = False)
			pyplot.show()

		theta_2_by_theta_1_distribution_array_array = numpy.swapaxes(numpy.swapaxes(theta_2_by_theta_1_distribution_array_array, 0, 1), 1, 2)
		
		pragmatic_receiver_signal_0_h_prior = numpy.sum(pragmatic_receiver(n-2)[0], axis = (0, 1))
		pragmatic_receiver_signal_1_h_prior = numpy.sum(pragmatic_receiver(n-2)[1], axis = (0, 1))
		pragmatic_receiver_signal_2_h_prior = numpy.sum(pragmatic_receiver(n-2)[2], axis = (0, 1))
		pragmatic_receiver_signal_not_1_h_prior = numpy.sum(pragmatic_receiver(n-2)[3], axis = (0, 1))
		pragmatic_receiver_signal_not_2_h_prior = numpy.sum(pragmatic_receiver(n-2)[4], axis = (0, 1))
		
		if n <= 2:
			pragmatic_receiver_signal_0_pre_normalized = pragmatic_sender(n-1)[0] * theta_2_by_theta_1_distribution_array_array[:,:,0:1] * pragmatic_receiver_signal_0_h_prior
			pragmatic_receiver_signal_1_pre_normalized = pragmatic_sender(n-1)[1] * theta_2_by_theta_1_distribution_array_array[:,:,1:2] * pragmatic_receiver_signal_0_h_prior
			pragmatic_receiver_signal_2_pre_normalized = pragmatic_sender(n-1)[2] * theta_2_by_theta_1_distribution_array_array[:,:,2:3] * pragmatic_receiver_signal_0_h_prior
			pragmatic_receiver_signal_not_1_pre_normalized = pragmatic_sender(n-1)[3] * theta_2_by_theta_1_distribution_array_array[:,:,3:4] * pragmatic_receiver_signal_0_h_prior
			pragmatic_receiver_signal_not_2_pre_normalized = pragmatic_sender(n-1)[4] * theta_2_by_theta_1_distribution_array_array[:,:,4:5] * pragmatic_receiver_signal_0_h_prior
		elif n > 2:
			pragmatic_receiver_signal_0_pre_normalized = pragmatic_sender(n-1)[0] * theta_2_by_theta_1_distribution_array_array[:,:,0:1] * pragmatic_receiver_signal_0_h_prior
			pragmatic_receiver_signal_1_pre_normalized = pragmatic_sender(n-1)[1] * theta_2_by_theta_1_distribution_array_array[:,:,1:2] * pragmatic_receiver_signal_1_h_prior
			pragmatic_receiver_signal_2_pre_normalized = pragmatic_sender(n-1)[2] * theta_2_by_theta_1_distribution_array_array[:,:,2:3] * pragmatic_receiver_signal_2_h_prior
			pragmatic_receiver_signal_not_1_pre_normalized = pragmatic_sender(n-1)[3] * theta_2_by_theta_1_distribution_array_array[:,:,3:4] * pragmatic_receiver_signal_not_1_h_prior
			pragmatic_receiver_signal_not_2_pre_normalized = pragmatic_sender(n-1)[4] * theta_2_by_theta_1_distribution_array_array[:,:,4:5] * pragmatic_receiver_signal_not_2_h_prior
		
		print 'level %s pragmatic_receiver_signal_0_pre_normalized = \n%s' % (n, pragmatic_receiver_signal_0_pre_normalized)
		print 'level %s pragmatic_receiver_signal_1_pre_normalized = \n%s' % (n, pragmatic_receiver_signal_1_pre_normalized)
		print 'level %s pragmatic_receiver_signal_2_pre_normalized = \n%s' % (n, pragmatic_receiver_signal_2_pre_normalized)
		print 'level %s pragmatic_receiver_signal_not_1_pre_normalized = \n%s' % (n, pragmatic_receiver_signal_not_1_pre_normalized)
		print 'level %s pragmatic_receiver_signal_not_2_pre_normalized = \n%s' % (n, pragmatic_receiver_signal_not_2_pre_normalized)
		
		pragmatic_receiver_signal_0_array = pragmatic_receiver_signal_0_pre_normalized / numpy.sum(pragmatic_receiver_signal_0_pre_normalized)
		pragmatic_receiver_signal_1_array = pragmatic_receiver_signal_1_pre_normalized / numpy.sum(pragmatic_receiver_signal_1_pre_normalized)
		pragmatic_receiver_signal_2_array = pragmatic_receiver_signal_2_pre_normalized / numpy.sum(pragmatic_receiver_signal_2_pre_normalized)
		pragmatic_receiver_signal_not_1_array = pragmatic_receiver_signal_not_1_pre_normalized / numpy.sum(pragmatic_receiver_signal_not_1_pre_normalized)
		pragmatic_receiver_signal_not_2_array = pragmatic_receiver_signal_not_2_pre_normalized / numpy.sum(pragmatic_receiver_signal_not_2_pre_normalized)
		
		pragmatic_receiver_memo[n] = numpy.asarray((pragmatic_receiver_signal_0_array, pragmatic_receiver_signal_1_array, pragmatic_receiver_signal_2_array, pragmatic_receiver_signal_not_1_array, pragmatic_receiver_signal_not_2_array))

		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 1, 2, 3)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 1, 2, 3)))

		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 1, 2)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 1, 2)))
		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 1, 3)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 1, 3)))
		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 2, 3)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 2, 3)))

		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 1)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 1)))
		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 2)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 2)))
		print 'numpy.sum(pragmatic_sender(%s-1), axis = (0, 3)) = %s' % (n, numpy.sum(pragmatic_sender(n-1), axis = (0, 3)))

		pragmatic_sender_signal_0_h_array = numpy.sum(numpy.sum(pragmatic_sender(n-1)[0] * theta_2_by_theta_1_distribution_array_array[:,:,0:1], axis = 0), axis = 0)
		pragmatic_sender_signal_1_h_array = numpy.sum(numpy.sum(pragmatic_sender(n-1)[1] * theta_2_by_theta_1_distribution_array_array[:,:,1:2], axis = 0), axis = 0)
		pragmatic_sender_signal_2_h_array = numpy.sum(numpy.sum(pragmatic_sender(n-1)[2] * theta_2_by_theta_1_distribution_array_array[:,:,2:3], axis = 0), axis = 0)
		pragmatic_sender_signal_not_1_h_array = numpy.sum(numpy.sum(pragmatic_sender(n-1)[3] * theta_2_by_theta_1_distribution_array_array[:,:,3:4], axis = 0), axis = 0)
		pragmatic_sender_not_signal_2_h_array = numpy.sum(numpy.sum(pragmatic_sender(n-1)[4] * theta_2_by_theta_1_distribution_array_array[:,:,4:5], axis = 0), axis = 0)

		if theta_distribution_relation == 'True':
	
			pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array = (pragmatic_sender(n-1)[0] * theta_2_by_theta_1_distribution_array_array[:,:,0:1])[fixed_theta_2_num, fixed_theta_1_num]
			print 'pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array
	
			pragmatic_sender_signal_1_fixed_theta_1_h_array = numpy.sum((pragmatic_sender(n-1)[1] * theta_2_by_theta_1_distribution_array_array[:,:,1:2])[:fixed_theta_1_num + 1, fixed_theta_1_num], axis = 0)
			pragmatic_sender_signal_1_fixed_theta_1_h_array = pragmatic_sender_signal_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array_array[:,:,1:2][:, fixed_theta_1_num], axis = 0)
			print 'pragmatic_sender_signal_1_fixed_theta_1_h_array = \n%s' % pragmatic_sender_signal_1_fixed_theta_1_h_array
	
			pragmatic_sender_signal_2_fixed_theta_2_h_array = numpy.sum((pragmatic_sender(n-1)[2] * theta_2_by_theta_1_distribution_array_array[:,:,2:3])[fixed_theta_2_num, fixed_theta_2_num:], axis = 0)
			pragmatic_sender_signal_2_fixed_theta_2_h_array = pragmatic_sender_signal_2_fixed_theta_2_h_array /  numpy.sum(theta_2_by_theta_1_distribution_array_array[:,:,2:3][fixed_theta_2_num, :], axis = 0)
			print 'pragmatic_sender_signal_2_fixed_theta_2_h_array = \n%s' % pragmatic_sender_signal_2_fixed_theta_2_h_array
	
			pragmatic_sender_signal_not_1_fixed_theta_1_h_array = numpy.sum((pragmatic_sender(n-1)[3] * theta_2_by_theta_1_distribution_array_array[:,:,3:4])[:fixed_theta_2_num + 1, fixed_theta_2_num], axis = 0)
			pragmatic_sender_signal_not_1_fixed_theta_1_h_array = pragmatic_sender_signal_not_1_fixed_theta_1_h_array / numpy.sum(theta_2_by_theta_1_distribution_array_array[:,:,3:4][:, fixed_theta_2_num], axis = 0)
			print 'pragmatic_sender_signal_not_1_fixed_theta_1_h_array = \n%s' % pragmatic_sender_signal_not_1_fixed_theta_1_h_array
	
			pragmatic_sender_not_signal_2_fixed_theta_2_h_array = numpy.sum((pragmatic_sender(n-1)[4] * theta_2_by_theta_1_distribution_array_array[:,:,4:5])[fixed_theta_1_num, fixed_theta_1_num:], axis = 0)
			pragmatic_sender_not_signal_2_fixed_theta_2_h_array = pragmatic_sender_not_signal_2_fixed_theta_2_h_array / numpy.sum(theta_2_by_theta_1_distribution_array_array[:,:,4:5][fixed_theta_1_num, :], axis = 0)
			print 'pragmatic_sender_not_signal_2_fixed_theta_2_h_array = \n%s' % pragmatic_sender_not_signal_2_fixed_theta_2_h_array
	
		elif theta_distribution_relation == 'False':
	
			pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array = pragmatic_sender(n-1)[0][fixed_theta_2_num, fixed_theta_1_num]
			print 'pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array = \n%s' % pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array
	
			pragmatic_sender_signal_1_fixed_theta_1_h_array = numpy.sum(pragmatic_sender(n-1)[1][:, fixed_theta_1_num], axis = 0)
			pragmatic_sender_signal_1_fixed_theta_1_h_array = pragmatic_sender_signal_1_fixed_theta_1_h_array / num_states
			print 'pragmatic_sender_signal_1_fixed_theta_1_h_array = \n%s' % pragmatic_sender_signal_1_fixed_theta_1_h_array
	
			pragmatic_sender_signal_2_fixed_theta_2_h_array = numpy.sum(pragmatic_sender(n-1)[2][fixed_theta_2_num, :], axis = 0)
			pragmatic_sender_signal_2_fixed_theta_2_h_array = pragmatic_sender_signal_2_fixed_theta_2_h_array / num_states
			print 'pragmatic_sender_signal_2_fixed_theta_2_h_array = \n%s' % pragmatic_sender_signal_2_fixed_theta_2_h_array
	
			pragmatic_sender_signal_not_1_fixed_theta_1_h_array = numpy.sum(pragmatic_sender(n-1)[3][:, fixed_theta_2_num], axis = 0)
			pragmatic_sender_signal_not_1_fixed_theta_1_h_array = pragmatic_sender_signal_not_1_fixed_theta_1_h_array / num_states
			print 'pragmatic_sender_signal_not_1_fixed_theta_1_h_array = \n%s' % pragmatic_sender_signal_not_1_fixed_theta_1_h_array
	
			pragmatic_sender_not_signal_2_fixed_theta_2_h_array = numpy.sum(pragmatic_sender(n-1)[4][fixed_theta_1_num, :], axis = 0)
			pragmatic_sender_not_signal_2_fixed_theta_2_h_array = pragmatic_sender_not_signal_2_fixed_theta_2_h_array / num_states
			print 'pragmatic_sender_not_signal_2_fixed_theta_2_h_array = \n%s' % pragmatic_sender_not_signal_2_fixed_theta_2_h_array

		pragmatic_receiver_signal_0_h_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][0], axis = 0), axis = 0)
		pragmatic_receiver_signal_0_h_array_densities = pragmatic_receiver_signal_0_h_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_0_h_array = \n%s' % pragmatic_receiver_signal_0_h_array
	
		pragmatic_receiver_signal_1_h_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][1], axis = 0), axis = 0)
		pragmatic_receiver_signal_1_h_array_densities = pragmatic_receiver_signal_1_h_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_1_h_array = \n%s' % pragmatic_receiver_signal_1_h_array
	
		pragmatic_receiver_signal_2_h_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][2], axis = 0), axis = 0)
		pragmatic_receiver_signal_2_h_array_densities = pragmatic_receiver_signal_2_h_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_2_h_array = \n%s' % pragmatic_receiver_signal_2_h_array
	
		pragmatic_receiver_signal_not_1_h_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][3], axis = 0), axis = 0)
		pragmatic_receiver_signal_not_1_h_array_densities = pragmatic_receiver_signal_not_1_h_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_not_1_h_array = \n%s' % pragmatic_receiver_signal_not_1_h_array
	
		pragmatic_receiver_not_signal_2_h_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][4], axis = 0), axis = 0)
		pragmatic_receiver_not_signal_2_h_array_densities = pragmatic_receiver_not_signal_2_h_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_not_signal_2_h_array = \n%s' % pragmatic_receiver_not_signal_2_h_array
	
		pragmatic_receiver_signal_1_theta_1_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][1], axis = 0), axis = 1)
		pragmatic_receiver_signal_1_theta_1_array_densities = pragmatic_receiver_signal_1_theta_1_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_1_theta_1_array = \n%s' % pragmatic_receiver_signal_1_theta_1_array
	
		pragmatic_receiver_signal_2_theta_2_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][2], axis = 1), axis = 1)
		pragmatic_receiver_signal_2_theta_2_array_densities = pragmatic_receiver_signal_2_theta_2_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_2_theta_2_array = \n%s' % pragmatic_receiver_signal_2_theta_2_array
	
		pragmatic_receiver_signal_not_1_theta_1_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][3], axis = 0), axis = 1)
		pragmatic_receiver_signal_not_1_theta_1_array_densities = pragmatic_receiver_signal_not_1_theta_1_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_signal_not_1_theta_1_array = \n%s' % pragmatic_receiver_signal_not_1_theta_1_array
	
		pragmatic_receiver_not_signal_2_theta_2_array = numpy.sum(numpy.sum(pragmatic_receiver_memo[n][4], axis = 1), axis = 1)
		pragmatic_receiver_not_signal_2_theta_2_array_densities = pragmatic_receiver_not_signal_2_theta_2_array / ((upper_bound - lower_bound)/num_states)
		print 'pragmatic_receiver_not_signal_2_theta_2_array = \n%s' % pragmatic_receiver_not_signal_2_theta_2_array

		fig, ax = pyplot.subplots(1, 2, figsize = (12,5))
	
		pyplot.subplot(1, 2, 1)
		line = pyplot.plot(array_0, pragmatic_sender_signal_0_h_array, lw = 2, color = 'k')
		line = pyplot.plot(array_0, pragmatic_sender_signal_1_h_array, lw = 2, color = 'b')
		line = pyplot.plot(array_0, pragmatic_sender_signal_2_h_array, lw = 2, color = 'r')
		line = pyplot.plot(array_0, pragmatic_sender_signal_not_1_h_array, lw = 2, color = 'c')
		line = pyplot.plot(array_0, pragmatic_sender_not_signal_2_h_array, lw = 2, color = 'orange')
	
		line = pyplot.plot(array_0, pragmatic_sender_signal_0_fixed_theta_1_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'k')
		line = pyplot.plot(array_0, pragmatic_sender_signal_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'b')
		line = pyplot.plot(array_0, pragmatic_sender_signal_2_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'r')
		line = pyplot.plot(array_0, pragmatic_sender_signal_not_1_fixed_theta_1_h_array, lw = 2, linestyle = '--', color = 'c')
		line = pyplot.plot(array_0, pragmatic_sender_not_signal_2_fixed_theta_2_h_array, lw = 2, linestyle = '--', color = 'orange')
	
		line = pyplot.plot(array_0, numpy.sum(pragmatic_sender(n-1)[1][:,:,fixed_theta_1_num], axis = 0) / (num_states), lw = 5, linestyle = ':', color = 'b')
		line = pyplot.plot(array_0, numpy.sum(pragmatic_sender(n-1)[2][:,:,fixed_theta_2_num], axis = 1) / (num_states), lw = 5, linestyle = ':', color = 'r')
		line = pyplot.plot(array_0, numpy.sum(pragmatic_sender(n-1)[3][:,:,fixed_theta_2_num], axis = 0) / (num_states), lw = 5, linestyle = ':', color = 'c')
		line = pyplot.plot(array_0, numpy.sum(pragmatic_sender(n-1)[4][:,:,fixed_theta_1_num], axis = 1) / (num_states), lw = 5, linestyle = ':', color = 'orange')
	
		pyplot.subplot(1, 2, 2)
		line = pyplot.plot(array_0, pragmatic_receiver_signal_0_h_array_densities, lw = 2, color = 'k')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_1_h_array_densities, lw = 2, color = 'b')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_1_theta_1_array_densities, lw = 2, linestyle = '--', color = 'b')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_2_h_array_densities, lw = 2, color = 'r')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_2_theta_2_array_densities, lw = 2, linestyle = '--', color = 'r')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_not_1_h_array_densities, lw = 2, color = 'c')
		line = pyplot.plot(array_0, pragmatic_receiver_signal_not_1_theta_1_array_densities, lw = 2, linestyle = '--', color = 'c')
		line = pyplot.plot(array_0, pragmatic_receiver_not_signal_2_h_array_densities, lw = 2, color = 'orange')
		line = pyplot.plot(array_0, pragmatic_receiver_not_signal_2_theta_2_array_densities, lw = 2, linestyle = '--', color = 'orange')
	
		pyplot.subplot(1, 2, 1)
		pyplot.legend([r'$\sigma_{%s}(u_{0}|h)$' % (n-1), r'$\sigma_{%s}(u_{1}|h)$' % (n-1), r'$\sigma_{%s}(u_{2}|h)$' % (n-1), r'$\sigma_{%s}(\neg u_{1}|h)$' % (n-1), r'$\sigma_{%s}(\neg u_{2}|h)$' % (n-1), r'$\sigma_{%s}(u_{0}|h, \theta_{1} \approx %s, \theta_{2} \approx %s)$' % ((n-1), numpy.around(array_0[fixed_theta_1_num], decimals = 2), numpy.around(array_0[fixed_theta_2_num], decimals = 2)), r'$\sigma_{%s}(u_{1}|h, \theta_{1} \approx %s)$' % ((n-1), numpy.around(array_0[fixed_theta_1_num], decimals = 2)), r'$\sigma_{%s}(u_{2}|h, \theta_{2} \approx %s)$' % ((n-1), numpy.around(array_0[fixed_theta_2_num], decimals = 2)), r'$\sigma_{%s}(\neg u_{1}|h, \theta_{1} \approx %s)$' % ((n-1), numpy.around(array_0[fixed_theta_2_num], decimals = 2)), r'$\sigma_{%s}(\neg u_{2}|h, \theta_{2} \approx %s)$' % ((n-1), numpy.around(array_0[fixed_theta_1_num], decimals = 2))], loc = 0, fontsize = 14)
	
		pyplot.subplot(1, 2, 2)
		pyplot.legend([r'$\rho_{%s}(h|u_{0})$' % n, r'$\rho_{%s}(h|u_{1})$' % n, r'$\rho_{%s}(\theta_{1}|u_{1})$' % n, r'$\rho_{%s}(h|u_{2})$' % n, r'$\rho_{%s}(\theta_{2}|u_{2})$' % n, r'$\rho_{%s}(h|\neg u_{1})$' % n, r'$\rho_{%s}(\theta_{1}|\neg u_{1})$' % n, r'$\rho_{%s}(h|\neg u_{2})$' % n, r'$\rho_{%s}(\theta_{2}|\neg u_{2})$' % n], loc = 0, fontsize = 14)

		fig.text(0, 0, r'$Lassiter\ and\ Goodman\ Three\ Signals\ with\ Not\ Iterated\ Arbitrary\ Variable\ Cost\ Variable\ Choice\ Parameter$' + '\n', fontsize = 8)
	
		fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost, decimals = 2)), str(numpy.around(cost_of_not, decimals = 2)), mu, sigma, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 		fig.text(.4, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost, decimals = 2)), str(numpy.around(cost_of_not, decimals = 2)), alpha_parameter, beta_parameter, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 		fig.text(.4, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost, decimals = 2)), str(numpy.around(cost_of_not, decimals = 2)), num_states, theta_distribution_type) + r'theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
	
		# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Iterated Arbitrary Normal Distribution.pdf')
		# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Iterated Arbitrary Beta Distribution.pdf')
		# pyplot.savefig('Lassiter and Goodman Model Three Signals with Not Iterated Arbitrary Uniform Distribution.pdf')
	
		pyplot.show()
		pyplot.close()

	return pragmatic_receiver_memo[n]

###################

# Here we have the settings for a level 0 receiver decoding probabilities, given a fixed
# theta. This forms the common basis for both Lassiter and Goodman's original model and
# our modified model.

cost_of_null_signal = 0.
cost_list = [1.]
cost_of_not_list = [1./3.]
choice_parameter_list = [4.]
lower_bound = -4.
upper_bound = 4.
num_states = 160

mu = 0.
sigma = 1.
state_distribution = norm(mu,sigma)

# alpha_parameter = 1.
# beta_parameter = 9.
# location_parameter = lower_bound
# scale_parameter = upper_bound - lower_bound
# state_distribution = beta(alpha_parameter, beta_parameter, loc = location_parameter, scale = scale_parameter)

# state_distribution = uniform(lower_bound, upper_bound - lower_bound)

fixed_theta_1_num = numpy.int(numpy.ceil(num_states*(8./12.)))
fixed_theta_2_num = numpy.int(numpy.ceil(num_states*(4./12.)))

theta_distribution_type = 'normal'
theta_distribution_relation = 'True'
theta_posterior_source = 'signal 1, not 1, signal 2, not 2'

if theta_distribution_type == 'normal':
	theta_1_distribution = norm(mu, sigma)
	theta_2_distribution = norm(mu, sigma)
elif theta_distribution_type == 'Beta':
	theta_1_distribution = beta(1, 9, loc = lower_bound, scale = upper_bound - lower_bound)
	theta_2_distribution = beta(1, 9, loc = lower_bound, scale = upper_bound - lower_bound)
elif theta_distribution_type == 'uniform':
	theta_1_distribution = uniform(lower_bound, upper_bound - lower_bound)
	theta_2_distribution = uniform(lower_bound, upper_bound - lower_bound)

array_0 = numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - ((numpy.flipud(numpy.linspace(upper_bound, lower_bound, num_states, endpoint = False)) - numpy.linspace(lower_bound, upper_bound, num_states, endpoint = False))/2)
print 'array_0 = %s' % array_0

pragmatic_sender_type = 'h_sensitive'
# pragmatic_sender_type = 'h_and_theta_sensitive'
# pragmatic_sender_type = 'h_sensitive_version_3'
# pragmatic_sender_type = 'modified h sensitive'

if pragmatic_sender_type == 'modified h sensitive':
	weighting_sigma = 1.

if pragmatic_sender_type == 'modified h sensitive':
	weighting_array = numpy.empty((0, num_states))
	for h_num in numpy.arange(num_states):
		weighting_array = numpy.insert(weighting_array, h_num, truncnorm.pdf(array_0, lower_bound, upper_bound, loc = array_0[h_num], scale = weighting_sigma), axis = 0)
	
max_level = 6

#########################

if theta_distribution_relation == 'True':
	initial_theta_1_on_theta_2_distribution_array = initial_theta_1_on_theta_2_distribution(theta_1_distribution, len(array_0))
	initial_theta_1_on_theta_2_distribution_array = initial_theta_1_on_theta_2_distribution_array / numpy.sum(initial_theta_1_on_theta_2_distribution_array, axis = 1)[numpy.newaxis].T
	initial_theta_1_on_theta_2_distribution_array = initial_theta_1_on_theta_2_distribution_array * numpy.transpose(theta_2_distribution.pdf(array_0)[numpy.newaxis])
	initial_theta_1_on_theta_2_distribution_array = initial_theta_1_on_theta_2_distribution_array / numpy.sum(initial_theta_1_on_theta_2_distribution_array)

	initial_theta_2_on_theta_1_distribution_array = initial_theta_2_on_theta_1_distribution(theta_2_distribution, len(array_0))
	initial_theta_2_on_theta_1_distribution_array = initial_theta_2_on_theta_1_distribution_array / numpy.sum(initial_theta_2_on_theta_1_distribution_array, axis = 0)
	initial_theta_2_on_theta_1_distribution_array = initial_theta_2_on_theta_1_distribution_array * theta_1_distribution.pdf(array_0)
	initial_theta_2_on_theta_1_distribution_array = initial_theta_2_on_theta_1_distribution_array / numpy.sum(initial_theta_2_on_theta_1_distribution_array)

	initial_theta_2_by_theta_1_distribution_array = (initial_theta_1_on_theta_2_distribution_array + initial_theta_2_on_theta_1_distribution_array)/2.

	initial_theta_2_by_theta_1_distribution_array = numpy.reshape(initial_theta_2_by_theta_1_distribution_array, [num_states, num_states, 1])
			
elif theta_distribution_relation == 'False':
	theta_1_distribution_array = theta_1_distribution.pdf(array_0)
	theta_1_distribution_array = theta_1_distribution_array/numpy.sum(theta_1_distribution_array)
	theta_2_distribution_array = theta_2_distribution.pdf(array_0)
	theta_2_distribution_array = theta_2_distribution_array/numpy.sum(theta_2_distribution_array)
	initial_theta_2_by_theta_1_distribution_array = theta_1_distribution_array * numpy.reshape(theta_2_distribution_array, (len(array_0), 1))
	initial_theta_2_by_theta_1_distribution_array = numpy.reshape(initial_theta_2_by_theta_1_distribution_array, [num_states, num_states, 1])

print 'initial_theta_2_by_theta_1_distribution_array = \n%s' % initial_theta_2_by_theta_1_distribution_array
print numpy.sum(initial_theta_2_by_theta_1_distribution_array)

plot_theta_1_distribution_array = numpy.sum(initial_theta_2_by_theta_1_distribution_array, axis = 0)
plot_theta_2_distribution_array = numpy.sum(initial_theta_2_by_theta_1_distribution_array, axis = 1)

fig, ax = pyplot.subplots(1,1)
pyplot.plot(array_0, plot_theta_1_distribution_array)
pyplot.plot(array_0, plot_theta_2_distribution_array)
pyplot.show()


#########################

receiver_0_signal_0_array = state_distribution.pdf(array_0)
receiver_0_signal_0_array = receiver_0_signal_0_array / numpy.sum(receiver_0_signal_0_array)
receiver_0_signal_0_array = numpy.tile(receiver_0_signal_0_array, (num_states, num_states, 1))
receiver_0_signal_0_array = receiver_0_signal_0_array * initial_theta_2_by_theta_1_distribution_array

receiver_0_signal_1_array = numpy.empty([0, num_states])
for theta_num in range(num_states):
	temp_signal_1_fixed_theta_array = numpy.empty(0)
	for h_num in range(num_states):
		value = receiver_0_signal_1(array_0[h_num], array_0[theta_num])
		temp_signal_1_fixed_theta_array = numpy.append(temp_signal_1_fixed_theta_array, value)
	receiver_0_signal_1_array = numpy.insert(receiver_0_signal_1_array, theta_num, temp_signal_1_fixed_theta_array, axis = 0)
receiver_0_signal_1_array = receiver_0_signal_1_array / numpy.tile(numpy.sum(receiver_0_signal_1_array, axis = 1), (num_states, 1)).T
receiver_0_signal_1_array = numpy.tile(receiver_0_signal_1_array, [num_states, 1, 1])
receiver_0_signal_1_array = receiver_0_signal_1_array * initial_theta_2_by_theta_1_distribution_array

receiver_0_signal_2_array = numpy.empty([0, num_states])
for theta_num in range(num_states):
	temp_signal_2_fixed_theta_array = numpy.empty(0)
	for h_num in range(num_states):
		value = receiver_0_signal_2(array_0[h_num], array_0[theta_num])
		temp_signal_2_fixed_theta_array = numpy.append(temp_signal_2_fixed_theta_array, value)
	receiver_0_signal_2_array = numpy.insert(receiver_0_signal_2_array, theta_num, temp_signal_2_fixed_theta_array, axis = 0)
receiver_0_signal_2_array = receiver_0_signal_2_array / numpy.tile(numpy.sum(receiver_0_signal_2_array, axis = 1), (num_states, 1)).T
receiver_0_signal_2_array = numpy.tile(numpy.reshape(receiver_0_signal_2_array, [num_states, 1, num_states]), [1, num_states, 1])
receiver_0_signal_2_array = receiver_0_signal_2_array * initial_theta_2_by_theta_1_distribution_array

receiver_0_signal_not_1_array = numpy.empty([0, num_states])
for theta_num in range(num_states):
	temp_signal_not_1_fixed_theta_array = numpy.empty(0)
	for h_num in range(num_states):
		value = receiver_0_signal_not_1(array_0[h_num], array_0[theta_num])
		temp_signal_not_1_fixed_theta_array = numpy.append(temp_signal_not_1_fixed_theta_array, value)
	receiver_0_signal_not_1_array = numpy.insert(receiver_0_signal_not_1_array, theta_num, temp_signal_not_1_fixed_theta_array, axis = 0)
receiver_0_signal_not_1_array = receiver_0_signal_not_1_array / numpy.tile(numpy.sum(receiver_0_signal_not_1_array, axis = 1), (num_states, 1)).T
receiver_0_signal_not_1_array = numpy.tile(receiver_0_signal_not_1_array, [num_states, 1, 1])
receiver_0_signal_not_1_array = receiver_0_signal_not_1_array * initial_theta_2_by_theta_1_distribution_array

receiver_0_signal_not_2_array = numpy.empty([0, num_states])
for theta_num in range(num_states):
	temp_signal_not_2_fixed_theta_array = numpy.empty(0)
	for h_num in range(num_states):
		value = receiver_0_signal_not_2(array_0[h_num], array_0[theta_num])
		temp_signal_not_2_fixed_theta_array = numpy.append(temp_signal_not_2_fixed_theta_array, value)
	receiver_0_signal_not_2_array = numpy.insert(receiver_0_signal_not_2_array, theta_num, temp_signal_not_2_fixed_theta_array, axis = 0)
receiver_0_signal_not_2_array = receiver_0_signal_not_2_array / numpy.tile(numpy.sum(receiver_0_signal_not_2_array, axis = 1), (num_states, 1)).T
receiver_0_signal_not_2_array = numpy.tile(numpy.reshape(receiver_0_signal_not_2_array, [num_states, 1, num_states]), [1, num_states, 1])
receiver_0_signal_not_2_array = receiver_0_signal_not_2_array * initial_theta_2_by_theta_1_distribution_array

for choice_parameter_num in numpy.arange(len(choice_parameter_list)):
	choice_parameter = choice_parameter_list[choice_parameter_num]
	for cost_num in numpy.arange(len(cost_list)):
		cost = cost_list[cost_num]
		cost_of_not = cost_of_not_list[cost_num]
		pragmatic_receiver_memo = {}
		pragmatic_sender_memo = {}
		pragmatic_receiver_memo[0] = numpy.asarray((receiver_0_signal_0_array, receiver_0_signal_1_array, receiver_0_signal_2_array, receiver_0_signal_not_1_array, receiver_0_signal_not_2_array))
		pragmatic_receiver(max_level)

choice_parameters_costs_pragmatic_receiver_h_array_densities = numpy.empty([0, len(cost_list), max_level/2 + 1, 5, num_states])
for choice_parameter_num in numpy.arange(len(choice_parameter_list)):
	costs_pragmatic_receiver_h_array_densities = numpy.empty([0, max_level/2 + 1, 5, num_states])
	for cost_num in numpy.arange(len(cost_list)):
		levels_pragmatic_receiver_h_array_densities = numpy.empty([0, 5, num_states])
		for pragmatic_receiver_level in sorted(pragmatic_receiver_memo):
			levels_pragmatic_receiver_h_array_densities = numpy.insert(levels_pragmatic_receiver_h_array_densities, pragmatic_receiver_level/2, numpy.sum(pragmatic_receiver_memo[pragmatic_receiver_level], axis = (1, 2))/((upper_bound - lower_bound)/num_states), axis =0)
		costs_pragmatic_receiver_h_array_densities = numpy.insert(costs_pragmatic_receiver_h_array_densities, cost_num, levels_pragmatic_receiver_h_array_densities, axis = 0)
	choice_parameters_costs_pragmatic_receiver_h_array_densities = numpy.insert(choice_parameters_costs_pragmatic_receiver_h_array_densities, choice_parameter_num, costs_pragmatic_receiver_h_array_densities, axis = 0)

for pragmatic_receiver_level in sorted(pragmatic_receiver_memo):

	fig, ax = pyplot.subplots(1, 1, figsize = (12,5))
	color_list = ['k', 'b', 'r', 'y', 'c']
	linestyle_list = ['-', '--']
	lw_list = [1, 3]
	pyplot.subplot(1, 1, 1)
	pyplot.grid(True)

	for choice_parameter_num in numpy.arange(len(choice_parameter_list)):
		for cost_num in numpy.arange(len(cost_list)):
			for signal in numpy.arange(5):
				pyplot.plot(array_0, choice_parameters_costs_pragmatic_receiver_h_array_densities[choice_parameter_num, cost_num, pragmatic_receiver_level/2, signal], color = color_list[signal], linestyle = linestyle_list[cost_num], lw = lw_list[choice_parameter_num])

	real_legend = pyplot.legend([r'$\rho_{%s}(h|u_{0})$' % pragmatic_receiver_level, r'$\rho_{%s}(h|u_{1})$' % pragmatic_receiver_level, r'$\rho_{%s}(h|u_{2})$' % pragmatic_receiver_level, r'$\rho_{%s}(h|\neg u_{1})$' % pragmatic_receiver_level, r'$\rho_{%s}(h|\neg u_{2})$' % pragmatic_receiver_level], loc = 'lower right', bbox_to_anchor = (.5, .5))
	for legobj in real_legend.legendHandles:
		legobj.set_linewidth(1.5)

	ax = pyplot.gca().add_artist(real_legend)
	dummy_line_0, = pyplot.plot([], label = r'$C(u) = 1/3 * length(u)$', color = 'gray', lw = 2)
	dummy_line_1, = pyplot.plot([], label = r'$C(u) = 4/3 * length(u)$', color = 'gray', lw = 2, linestyle = '--')
	dummy_line_2, = pyplot.plot([], label = r'$\lambda = 4.$', color = 'gray', lw = 1)
	dummy_line_3, = pyplot.plot([], label = r'$\lambda = 8.$', color = 'gray', lw = 3)
	pyplot.legend(handles = [dummy_line_0, dummy_line_1, dummy_line_2, dummy_line_3], loc = 'lower left', bbox_to_anchor = (.5, .5), fontsize = 12)

	fig.text(0, 0, r'$Lassiter\ and\ Goodman\ Three\ Signals\ with\ Not\ Iterated\ Arbitrary\ Variable\ Cost\ Variable\ Choice\ Parameter$' + '\n', fontsize = 8)

	fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter_list, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list, decimals = 2)), str(numpy.around(cost_of_not_list, decimals = 2)), mu, sigma, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 	fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list, decimals = 2)), str(numpy.around(cost_of_not_list, decimals = 2)), alpha_parameter, beta_parameter, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 	fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list, decimals = 2)), str(numpy.around(cost_of_not_list, decimals = 2)), num_states, theta_distribution_type) + r'theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)

	pyplot.show()
	pyplot.close()

for choice_parameter_num in numpy.arange(len(choice_parameter_list)):
	for cost_num in numpy.arange(len(cost_list)):
	
# 		fig, ax = pyplot.subplots(1, 1, figsize = (12,5))
		fig = pyplot.figure(figsize = (12,5))
		color_list = ['k', 'b', 'r', 'y', 'c']
		linestyle_list = ['-', '--']
		lw_list = [1., 3.]
# 		pyplot.subplot(1, 1, 1)
		ax = fig.add_subplot(111)
		pyplot.grid(True)
		ax.set_ylim(0., 1.2)

		for pragmatic_receiver_level in sorted(pragmatic_receiver_memo):
			for signal in numpy.arange(5):
				pyplot.plot(array_0, choice_parameters_costs_pragmatic_receiver_h_array_densities[choice_parameter_num, cost_num, pragmatic_receiver_level/2, signal], color = color_list[signal], linestyle = linestyle_list[cost_num], lw = pragmatic_receiver_level/2 + 1)

		ax.text(.5, .8, r'$\lambda = %s, C(u_{0}) \approx %s,$' % (choice_parameter_list[choice_parameter_num], str(numpy.around(cost_of_null_signal, decimals = 2))) + '\n' + r'$C(u_{n}) \approx %s, C(\neg u_{n}) \approx %s$' % (str(numpy.around(cost_list[cost_num], decimals = 2)), str(numpy.around(cost_list[cost_num] + cost_of_not_list[cost_num], decimals = 2))), bbox={'facecolor':'white', 'alpha':1., 'pad':10}, horizontalalignment = 'center', verticalalignment = 'center', transform = ax.transAxes)

		real_legend = pyplot.legend([r'$\rho_{n}(h|u_{0})$', r'$\rho_{n}(h|u_{1})$', r'$\rho_{n}(h|u_{2})$', r'$\rho_{n}(h|\neg u_{1})$', r'$\rho_{n}(h|\neg u_{2})$'], loc = 'upper left', bbox_to_anchor = (.025, .975))
		for legobj in real_legend.legendHandles:
			legobj.set_linewidth(1.5)
		ax = pyplot.gca().add_artist(real_legend)
		
		dummy_line_list = []
		for pragmatic_receiver_level in sorted(pragmatic_receiver_memo):
			dummy_line, = pyplot.plot([], label = r'$\rho_{%s}$' % pragmatic_receiver_level, color = 'gray', lw = pragmatic_receiver_level/2 + 1)
			dummy_line_list.append(dummy_line)
		pyplot.legend(handles = dummy_line_list, loc = 'upper right', bbox_to_anchor = (.975, .975), fontsize = 14)

# 		dummy_line_0, = pyplot.plot([], label = r'$C(u) = 1/3 * length(u)$', color = 'gray', lw = 2)
# 		dummy_line_1, = pyplot.plot([], label = r'$C(u) = 4/3 * length(u)$', color = 'gray', lw = 2, linestyle = '--')
# 		dummy_line_0, = pyplot.plot([], label = r'$\rho_{0}$', color = 'gray', lw = 1)
# 		dummy_line_1, = pyplot.plot([], label = r'$\rho_{2}$', color = 'gray', lw = 2)
# 		dummy_line_2, = pyplot.plot([], label = r'$\rho_{4}$', color = 'gray', lw = 3)
# 		dummy_line_3, = pyplot.plot([], label = r'$\rho_{6}$', color = 'gray', lw = 4)
# 		pyplot.legend(handles = [dummy_line_0, dummy_line_1, dummy_line_2, dummy_line_3], loc = 'upper right', bbox_to_anchor = (.975, .975), fontsize = 14)

		fig.text(0, 0, r'$Lassiter\ and\ Goodman\ Three\ Signals\ with\ Not\ Iterated\ Arbitrary\ Variable\ Cost\ Variable\ Choice\ Parameter$' + '\n', fontsize = 8)

		fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \mu = %s, \sigma = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter_list[choice_parameter_num], str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list[cost_num], decimals = 2)), str(numpy.around(cost_of_not_list[cost_num], decimals = 2)), mu, sigma, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 		fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, \alpha = %s, \beta = %s, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list, decimals = 2)), str(numpy.around(cost_of_not_list, decimals = 2)), alpha_parameter, beta_parameter, num_states, theta_distribution_type) + r'$theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)
# 		fig.text(0, 0, r'$\lambda = %s, C(u_{0}) \approx %s, C(u_{n}) \approx %s, C(\neg) \approx %s, Uniform distribution, num\ states = %s, theta\ distribution\ type = %s,$' % (choice_parameter, str(numpy.around(cost_of_null_signal, decimals = 2)), str(numpy.around(cost_list, decimals = 2)), str(numpy.around(cost_of_not_list, decimals = 2)), num_states, theta_distribution_type) + r'theta\ distribution\ relation = %s, theta\ posterior\ source = %s, pragmatic\ sender\ type = %s$' % (theta_distribution_relation, theta_posterior_source, pragmatic_sender_type), fontsize = 8)

		pyplot.show()
		pyplot.close()
