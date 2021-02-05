#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import matplotlib.pyplot as plt

class noise_gate_sigma:
	def __init__(self, bg_file, octave):
		nde_class = nde.decompose(bg_file)
		nde_class.octaves = octave
		nde_class.decompose('ns~background')
	
		fp = h5py.File('ns~background.hdf5', 'r', libver='latest')
		
		self.bg_sigma = {}
		self.bg_mean = {}
		
		for key in list(fp.keys()):
			
			if key == 'meta':
				continue
			
			self.bg_sigma[key] = np.std(self.wave2vol(fp[key], spread=1000))
			self.bg_mean[key] = np.mean(self.wave2vol(fp[key], spread=1000))
			
		fp.close()
		#x = np.linspace(0,1,len(self.bg_mean.values()))
		#y = np.array([*self.bg_mean.values()])
		#e = np.array([*self.bg_sigma.values()])
		
		#print(x, y, e)
		
		#plt.errorbar(x, y, e)
		#plt.show()
		
		return

	def wave2vol(self, wave, spread=10, detect_type='peak'):
		
		oldlen = len(wave)
		newlen = int(np.ceil(len(wave)/spread))*spread
		
		wave = np.append(wave, np.zeros(newlen - oldlen))
		
		wave = wave.reshape(int(newlen/spread), spread)
		
		if detect_type == 'peak':
			volume = np.max(np.abs(wave), axis=1)
		elif detect_type == 'rms':
			volume = np.sqrt(np.sum(wave**2, axis=1) / wave.shape[1])
		else:
			raise Exception("only peak and rms detection types are defined so far")
		
		volume = np.repeat(volume, spread)
		
		volume = volume[0:oldlen]
		
		return volume
	
	def noisegate(self, data, key, sigma, spread=10):
		
		""" noise gate: let noise through once it reaches a certain level
		level => abs level at witch the noise gate activates
		data => the data we want to operate on
		mult => scalar multiplier for the data
		spread => how long after the last activation do we shut off
		
		return: gated data
		side effects: None
		"""
		
		level = self.bg_mean[key] + (self.bg_sigma[key] * sigma)
		
		# calculate the locations where the data is greater than level
		gated_data_mask = np.greater(np.abs(data), level)
		
		volume = self.wave2vol(gated_data_mask, spread)
		
		# set selected cells to 0
		data[np.logical_not(volume)] = 0
		
		return data 
