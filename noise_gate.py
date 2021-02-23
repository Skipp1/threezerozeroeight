#! /usr/bin/env python3

import numpy as np
import note_utils as note
import h5py
import note_decompose as nde
import scipy.signal as signal


def wave2vol(data=None, spread=None, detect_type='peak'):
	""" convert AC audio into DC volume, do this by dicing up the audio and picking
	the peak or the rms value as the audio
	
	spread => the size of the window
	detect_type => how we are finding the volume
	wave => tuple containing fp & key, or the actual data itself
	
	return:
	volume array of same len as wave
	
	side effects:
	None
	"""
	
	if type(data) == tuple:
		fp_key = True
		fp = data[0]
		key = data[1]
		wave = np.copy(fp[key])
	
	else:
		fp_key = False
		wave = np.copy(data)
	
	if spread is None:
		if fp_key:
			# (Lin et al. 2008 Varying-Window-Length Time-Frequency Peak Filtering And Its Application
			# To Seismic Data): The optimal Window Length can be expressed as a function of the dominant
			# frequency fd and the sampling frequency fs
			fd = note.note2freq(int(key.split('-')[0]), key.split('-')[1])
			fs = fp['meta'][1]
			
			alpha = 0.384  # Lin et al 2008
			
			spread = alpha * fs / fd
			spread = int(np.ceil(spread))
		
		else:
			spread = 1000  # default value if we dont have spread
	
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
	""" class only used to store the mean & sigma global vars """
	def __init__(self, bg_file, octave):
		""" calculate & store the mean and stddev values for a short sample of the 
		background noise
		
		input:
		bg_file => path to background file
		octave -> octaves that we are looking at (needs to be the same as the main audio)
		
		return:
		None
		
		Side effects:
		create file for background noise &  
		update self.bg_sigma and self.bg_mean
		"""
		
		nde_class = nde.decompose(bg_file)
		nde_class.octaves = octave
		nde_class.decompose('ns~background')
		
		fp = h5py.File('ns~background.hdf5', 'r', libver='latest')
		
		self.bg_sigma = {}
		self.bg_mean = {}
		
		for key in list(fp.keys()):
			
			if key == 'meta':
				continue
			
			self.bg_sigma[key] = np.std(wave2vol((fp, key)))
			self.bg_mean[key] = np.mean(wave2vol((fp, key)))
		
		fp.close()
		return
	
	def noise_gate_sigma(self, data, key, sigma, spread=None):
		""" noise gate: let noise through once it reaches a certain level
		data => the data we want to operate on
		sigma => stdev above the mean we want to look at
		spread => how long after the last activation do we shut off
		key => what note we are looking at
		
		return: gated data
		side effects: None
		"""
		
		if type(data) == h5py._hl.files.File:
			volume = wave2vol((data, key), spread=spread)
			data = data[key]
		else:
			volume = wave2vol(data, spread=spread)
		
		level = self.bg_mean[key] + (self.bg_sigma[key] * sigma)
		
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
