#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import scipy.signal as signal
import matplotlib.pyplot as plt



def wave2vol(wave, spread=10, detect_type='peak'):
	
	oldlen = len(wave)
	newlen = int(np.ceil(len(wave) / spread)) * spread
	
	wave = np.append(wave, np.zeros(newlen - oldlen))
	
	wave = wave.reshape(int(newlen / spread), spread)
	
	if detect_type == 'peak':
		volume = np.max(np.abs(wave), axis=1)
	elif detect_type == 'rms':
		volume = np.sqrt(np.sum(wave**2, axis=1) / wave.shape[1])
	else:
		raise Exception("only peak and rms detection types are defined so far")
	
	volume = np.repeat(volume, spread)
	
	volume = volume[0:oldlen]
	
	return volume


# -------------------------------------------------------------------------------------------- #


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
			
			self.bg_sigma[key] = np.std(wave2vol(fp[key], spread=1000))
			self.bg_mean[key] = np.mean(wave2vol(fp[key], spread=1000))
		
		fp.close()
		return
	
	def noise_gate_sigma(self, data, key, sigma, spread=1000):
		""" noise gate: let noise through once it reaches a certain level
		level => abs level at witch the noise gate activates
		data => the data we want to operate on
		mult => scalar multiplier for the data
		spread => how long after the last activation do we shut off
		
		return: gated data
		side effects: None
		"""
		
		level = self.bg_mean[key] + (self.bg_sigma[key] * sigma)
		
		volume = wave2vol(data, spread=spread)
		
		vol_mask = np.logical_not(np.greater(volume, level))
		
		new_data = np.copy(data)
		new_data[vol_mask] = 0
		
		return new_data


# -------------------------------------------------------------------------------------------- #

try:
	# bit of an esoteric library, so dont error if not found
	import tftb.processing as tftb
except:
	print("tftb not found")
	tftb = False

def noise_gate_PWVD(data, spread=1000):
	
	# TODO something smoothed_pseudo_wigner_ville
	if tbtb is False:
		raise FileNotFoundError("tftb must be available to use PWVD")
	
	data_volume = wave2vol(data, spread=spread)
	
	# ------- PWVD ---------------------------- #
	fwindow = signal.hamming(1)
	twindow = signal.hamming(1)
	
	de_dup_vol = data_volume[0::spread]
	
	spec = tftb.smoothed_pseudo_wigner_ville(de_dup_vol, fwindow=fwindow, twindow=twindow)
	m = np.max(spec, axis=0)
	
	ada_volume = m * np.max(de_dup_vol)
	
	ada_volume = np.repeat(ada_volume, spread)
	
	ada_volume = ada_volume[0:len(volume)]
	# -----------------------------------------#
	
	volume_scale = ada_volume / data_volume
	
	new_data = np.copy(data)
	new_data = new_data * volume_scale
	
	return new_data
