#! /usr/bin/env python3

import numpy as np
#import h5py


#def spreading(data, spread):
	
	#for i in range(len(data)):
		#if (i < spread):
			#segment = np.zeros(spread)
			#segment[-i-1:-1] = data[0:i]
		#else:
			#segment = data[i-spread:i]
			
		#if segment.all() > 1:
			#data[i] = True
		

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
	newlen = int(np.floor(len(data)/spread))*spread
	data = data[0:newlen]
	gated_data_mask = np.greater(np.abs(data), level)
	
	#spreading(gated_data_mask, spread)
	gated_data_mask = gated_data_mask.reshape(int(newlen/spread), spread)
	gated_data_mask = np.sum(gated_data_mask, axis=1, dtype=bool)
	
	# TODO: numpy repeat
	gated_data_mask = np.repeat(gated_data_mask, spread)
	
	# set selected cells to 0
	data[np.logical_not(gated_data_mask)] = 0
	
	return data 


def noisegate_rel(self, level, data, spread=10):
	""" same as above but use a level between 0-1 
	Returns the relative noisegate
	
	NOTE: If the note is not found anywhere, it will
	accidentially pick stuff up from random noisegate
	"""
	
	data = data / np.max(data)
	return noisegate(level, data, spread)


#def noisegate_deriv(grad, data, sample_rate):
	##TODO
	#""" do something with the gradient? (IDK havent implemented it yet)
	#"""
	
	#gradient_data = np.gradient(data, 1/sample_rate)
	#gated_data = gradient_data > grad 
	
	#return gated_data 

