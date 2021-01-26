#! /usr/bin/env python3

import numpy as np
from scipy.linalg import blas
import h5py

#class noisegate:
	
	#def __init__(self):
		#""" load init global prams """
		#try:
			##check to see if file exists
			#self.fp = h5py.File('i_roll.hdf5', 'r', libver='latest')
			#self.size = self.fp['size']
			
		#except OSError:
			## create a dummy file if it doesnt
			#self.update_iroll(1, 1, init=True)
			
		#return 
	
	#def update_iroll(self, size, spread, init=False):
		
		#""" Because the matrix required for spreading the data takes quite a while to generate, we want to call this function as new times as possible, as such 
		#we save the result from the computation and try and use the cached result 
		
		#inputs:
		#size => size of the array we are creating
		#spread => how far we want to spread
		#init => are we creating the file from scratch
		
		#global inputs:
		#fp => hdf file 
		
		#return: new i_roll
		#side effects: i_roll.hdf5 gets updated
		              #self.size gets updated 
		#"""
		
		## close the read only mode, if the file is already open 
		#if not init: 
			#self.fp.close()
		
		## open file in write mode
		#self.fp = h5py.File('i_roll.hdf5', 'w', libver='latest')
		
		#i_roll = np.eye(size, dtype=bool, order='F') # identity
		#for i in range(1, spread+1):
			#i_roll = np.logical_or(i_roll, np.eye(size, k= i, dtype=bool, order='F'))
			
		## save the newly created data 
		#self.fp.create_dataset('i_roll', data=i_roll, dtype=bool)
		#self.fp.create_dataset('size', data=size, dtype=int)

		## update the global var 
		#self.size = size

		## change file back to read only 
		#self.fp.close()
		#self.fp = h5py.File('i_roll.hdf5', 'r', libver='latest')
		
		#return i_roll
	
	
	#def spreading(self, data, spread):
		
		#size = len(data)
		
		#if size != self.size:
			#i_roll = self.update_iroll(size, spread)
		#else:
			#i_roll = self.fp['i_roll']
		
		## sgemv => blas function for a(M V)
		## where a => float,  M => matrix, V=> vector 
		#return np.equal(blas.sgemv(1.0, i_roll, data), 0

def spreading(data, spread):
	i_roll = np.eye(spread, k=1, dtype=bool, order='F')
	
	for i in range(len(data)):
		if (i < spread):
			segment = np.zeros(spread)
			segment[-i-1:-1] = data[0:i]
		else:
			segment = data[i-spread:i]
	
	if np.sum(blas.sgemv(1.0, i_roll, segment)) > 1:
		data[i] = True

def noisegate(level, data, mult=1, spread=10):
	
	""" noise gate: let noise through once it reaches a certain level
	level => abs level at witch the noise gate activates
	data => the data we want to operate on
	mult => scalar multiplier for the data
	spread => how long after the last activation do we shut off
	
	return: gated data
	side effects: None
	"""
	# calculate the opening of the gate
	gated_data_mask = np.greater(data, level)
	
	data = np.copy(data)
	
	# 'smear' the mask so we select cells after the peak
	gated_data_mask = spreading(gated_data_mask, spread)
	
	# set selected cells to 0
	data[gated_data_mask] = 0
	
	return data 


def noisegate_rel(self, level, data, spread=10):
	""" same as above but use a level between 0-1 
	Returns the relative noisegate
	
	NOTE: If the note is not found anywhere, it will
	accidentially pick stuff up from random noisegate
	"""
	
	data = data / np.max(data)
	return noisegate(level, data, spread)


def noisegate_deriv(grad, data, sample_rate):
	#TODO
	""" do something with the gradient? (IDK havent implemented it yet)
	"""
	
	gradient_data = np.gradient(data, 1/sample_rate)
	gated_data = gradient_data > grad 
	
	return gated_data 

