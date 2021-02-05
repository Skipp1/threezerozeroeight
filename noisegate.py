#! /usr/bin/env python3

import numpy as np

def noisegate(level, data, mult=1, spread=10):
	
	""" noise gate: let noise through once it reaches a certain level
	level => abs level at witch the noise gate activates
	data => the data we want to operate on
	mult => scalar multiplier for the data
	spread => how long after the last activation do we shut off
	
	return: gated data
	side effects: None
	"""
	
	# trim the end of the data to make it a multiple of spread
	oldlen = len(data)
	newlen = int(np.floor(len(data)/spread))*spread
	
	# trim the data 
	data = data[0:newlen]
	
	# calculate the locations where the data is greater than level
	gated_data_mask = np.greater(np.abs(data), level)
	
	# transform into a 2d array of shape spread x newlen/spread
	gated_data_mask = gated_data_mask.reshape(int(newlen/spread), spread)
	
	# sum along the spread axis
	gated_data_mask = np.sum(gated_data_mask, axis=1, dtype=bool)
	
	# return to newlen, by repeating each value spread times
	gated_data_mask = np.repeat(gated_data_mask, spread)
	
	# set selected cells to 0
	data[np.logical_not(gated_data_mask)] = 0
	
	# restore the data to its orig length to prevent confusion later on
	data = np.append(data, np.zeros(oldlen - newlen))
	
	return data 


def noisegate_rel(level, data, spread=10):
	""" same as above but use a level between 0-1 
	Returns the relative noisegate
	
	NOTE: If the note is not found anywhere, it will
	accidentially pick stuff up from random noisegate
	"""
	# convert to a 1-0 scale
	data_max = np.max(data) 
	data = data / data_max
	
	# apply noisegate
	gated_data = noisegate(level, data, spread)
	
	# return to previous scaling
	return gated_data * data_max


