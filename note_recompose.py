#! /usr/bin/env python3

import numpy as np
from scipy.io import wavfile
import fft_wrapper as fft
import h5py
import os

def save_prep(key_data, n_cpu, save_type):
	""" this fuction preps the data for saving to a wav file 
	This is only needed in case the data is save_type = 1
	
	inputs:
	key_data => data from the hdf5 file
	n_cpu => cpu cores to use when using pyfftw
	save_type => was the data saved in real or fourier form
	
	outputs:
	return => data in real form
	side_effects => None
	"""
	
	if save_type == 0:
		# saved in real form, just return
		return key_data
	
	elif save_type == 1:
		# saved in fourier form, convert to real & return
		return fft.ifft(key_data, threads=n_cpu, overwrite_input=True)
	
	else:
		# something horrid happened
		raise Exception("No idea how to handle save_type" + save_type)

def recompose(in_file, out_file='out.wav', compose_to='combine'):
	
	""" recompose a file from a bunch of diffrent decomposed notes 
	NOTE: not in the decompose class, but resides in the same file
	
	inputs: in_file, hdf5 file we want to recompose
	        out_file, wav file we want to write to
	
	return: None
	side_effects: generate wav file in filesystem
	"""
	
	fp = h5py.File(in_file + '.hdf5', 'r', libver='latest')
	
	# metadata key 
	save_type, sample_rate, file_len, fourier_len = fp['meta']

	# init some variables
	n_cpu = os.cpu_count()	
	data = np.zeros(file_len)

	for key in list(fp.keys()):
		# skip over the meta key 
		if key == 'meta':
			continue
		
		if compose_to == 'combine':
			# convert to real & sum
			data += save_prep(np.real(fp[key]) + np.imag(fp[key]), n_cpu, save_type)
			
		else:
			# save each key as a seperate wav file depending on the note
			save_data = save_prep(fp[key], n_cpu, save_type)	
			save_data = (np.real(save_data) + np.imag(save_data))
			wavfile.write(key + '.wav', sample_rate, save_data)
		
	if compose_to == 'combine':
		# save the summation of the data
		# TODO fix scaling
		save_data = (np.real(data) + np.imag(data)) / (440*len(fp.keys())-1)
		wavfile.write(out_file, sample_rate, save_data)
		
	return 

# ------------------------------------------------------------------------------------------------- #

def cmdline_recomp():
	""" for use on the command line """
	file_in = input("what file should I recompose: ") 
	compose_type = input("all [y/n]: ") 
	if compose_type.upper() == "Y":
		recompose(file_in, compose_to='all')
	else:
		recompose(file_in, out_file="recomposed.wav")

if __name__ == "__main__":
	cmdline_recomp()
