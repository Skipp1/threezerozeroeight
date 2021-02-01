#! /usr/bin/env python3

import numpy as np
from scipy.io import wavfile
import pyfftw
import h5py
import os

def recompose(in_file, out_file):
	
	""" recompose a file from a bunch of diffrent decomposed notes 
	NOTE: not in the decompose class, but resides in the same file
	
	inputs: in_file, hdf5 file we want to recompose
	        out_file, wav file we want to write to
	
	return: None
	side_effects: generate wav file in filesystem
	"""
	
	fp = h5py.File(in_file + '.hdf5', 'r', libver='latest')
	
	savetype, sample_rate, len_filedata, len_fourier_data = fp['meta']
	
	#TODO fix this hack
	spread = 1000
	fourier_data = np.zeros(int(np.floor(len_filedata/spread))*spread)
	
	n_cpu = os.cpu_count()

	
	if savetype == 0:
		for key in list(fp.keys()):
			
			# skip over the meta key 
			if key == 'meta':
				continue
				# I assume we can just sum them?
			fourier_data += fp[key]
		
	elif savetype == 1:
		for key in list(fp.keys()):
			# skip over the meta key 
			if key == 'meta':
				continue
			# again just summing
			fourier_data += pyfftw.interfaces.numpy_fft.ifft(fourier_data / (len(fp.keys())-1),
																									threads=n_cpu, overwrite_input=True)
		
	else:
		 raise Exception("No idea how to handle savetype" + savetype)
		
	save_data = fourier_data / (len(fp.keys())-1)
	
	save_data = np.real(save_data) + np.imag(save_data)
	
	wavfile.write(out_file + '.wav', sample_rate, save_data)
	
	return 

def split2wav(file_in):
		
	""" recompose a file from a bunch of diffrent decomposed notes 
	NOTE: not in the decompose class, but resides in the same file
	
	inputs: in_file, hdf5 file we want to recompose
	        out_file, wav file we want to write to
	
	return: None
	side_effects: generate wav file in filesystem
	"""
	
	fp = h5py.File(in_file + '.hdf5', 'r', libver='latest')
	
	savetype, sample_rate, len_filedata, len_fourier_data = fp['meta']
	
	#TODO fix this hack
	spread = 1000
	fourier_data = np.zeros(int(np.floor(len_filedata/spread))*spread)
	
	n_cpu = os.cpu_count()

	if savetype == 0:
		for key in list(fp.keys()):
			
			# skip over the meta key 
			if key == 'meta':
				continue
				# I assume we can just sum them?
			save_data = fp[key]
			save_data = np.real(save_data) + np.imag(save_data)
			wavfile.write(key + '.wav', sample_rate, save_data)
		
	elif savetype == 1:
		for key in list(fp.keys()):
		
			# skip over the meta key 
			if key == 'meta':
				continue
			save_data = pyfftw.interfaces.numpy_fft.ifft(fourier_data / (len(fp.keys())-1),
																								threads=n_cpu, overwrite_input=True)
			save_data = np.real(save_data) + np.imag(save_data)
			wavfile.write(key + '.wav', sample_rate, save_data)
	
	else:
		 raise Exception("No idea how to handle savetype" + savetype)
		
	return 


# ------------------------------------------------------------------------------------------------- #
def cmdline_recomp():
	file_in = input("what file should I recompose: ") 
	recompose(file_in, "recomposed")

if __name__ == "__main__":
	cmdline_recomp()
