#! /usr/bin/env python3

import numpy as np
from scipy.linalg import blas
import math_fun

def spreading(data, spread, hist):
	
	size = len(data)
	
	i_roll = np.eye(size, dtype=bool, order='F') # identity
	for i in range(1, spread+hist+1):
		i_roll = np.logical_or(i_roll, np.eye(size, k= i, dtype=bool, order='F')) # upper shift
	for i in range(1, spread+1):
		i_roll = np.logical_or(i_roll, np.eye(size, k=-i, dtype=bool, order='F')) # lower shift
		
	return np.equal(blas.ssymv(1.0, i_roll, data), 0)


def noisegate(level, data, spread=10, hist=0):
	
	data = np.asfortranarray(data)
	
	gated_data_mask = np.greater(data, level)
	
	gated_data_mask = spreading(gated_data_mask, spread, hist)
	
	data[gated_data_mask] = 0
	
	return data


def noisegate_rel(self, level, data, spread=10, hist=0):
	""" same as above but use a level between 0-1 
	Returns the relative noisegate
	
	NOTE: If the note is not found anywhere, it will
	accidentially pick stuff up from random noisegate
	"""
	
	data = data / np.max(data)
	return noisegate(level, data, spread, hist)


def noisegate_deriv(self, grad, data, sample_rate):
	
	gradient_data = np.gradient(data, 1/sample_rate)
	gated_data = gradient_data > grad 
	
	return gated_data 

