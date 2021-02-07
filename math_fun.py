#! /usr/bin/env python3
import numpy as np

""" common maths functions that are not provided by default"""

def gaussian(x, mu, sig):
	""" return a Gaussian distribution over x
	inputs:
		x => x axis data of distribution
		mu => the mean value of data
		sig => standard deviation of data
	returns: Gaussian distribution 
	"""
	return 1/(sig * np.sqrt(2*np.pi)) * np.exp((-1/2) * ((x - mu)/sig)**2)

def gaussian_max1(x, mu, sig):
	""" return a Gaussian distribution over x where the pean of the gaussian is 1
	inputs:
		x => x axis data of distribution
		mu => the mean value of data
		sig => standard deviation of data
	returns: Gaussian distribution with a peak of 1
	"""
	return np.exp((-1/2) * ((x - mu)/sig)**2)

def pink_power(f, alpha=1, scale=1):
	""" power spectral density for pink noise 
	inputs:
		f => frequency
		alpha => exponent characteristic to pink noise
		scale => scaling factor of proportionality
	returns: power spectral density"""
	return scale / f**alpha
