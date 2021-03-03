#! /usr/bin/env python3

import numpy as np
import note_utils as note
import h5py
import note_decompose as nde
import scipy.signal as signal
import adaptfilt as ada


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


class noise_gate_adaptave:
	
	# not actually a noise gate but whatever
	
	def __init__(self, bg_file, octave):
		# decompose the background noise
		nde_class = nde.decompose(bg_file)
		nde_class.octaves = octave
		nde_class.decompose('ns~background')
		
		# ---- and now for global vars in disguise ----- #
		# file stuff
		self.fp = h5py.File('ns~background.hdf5', 'r', libver='latest')
		self.spread = 1000
		self.data_len = 0
		
		# vars for nlms
		self.taps = 100
		self.mu = 0.1
		self.eps = 0.001
		self.leak = 0
		self.initCoeffs = None
		self.n = None
		self.returnCoeffs = False
		
		# other interesting values that get generated
		self.err = 0
		self.weights = 0
		# --------------------------------------------- #
		
		return
	
	def dedup_vol(self, vol):
		""" wave2vol returns a repeated thing 
		1 1 1 2 2 2 3 3 3 etc 
		de-duplicate that
		
		input: volume 
		
		global input: spread => amount of times each value is duplicated
		
		output: shortened volume
		side_effects: None
		"""
		
		return vol[0::self.spread]
	
	def redup_vol(self, vol):
		""" wave2vol returns a repeated thing 
		1 1 1 2 2 2 3 3 3 etc 
		dedup_vol deduplicates that
		
		set us re-duplicate that to its init length
		
		input: volume 
		
		global input: spread => amount of times each value is duplicated
		              data_len => how long the data was origenally
		
		output: re-lengthened volume
		side_effects: None
		"""
		
		return np.repeat(vol, self.spread)[:self.data_len]
	
	def noise_gate_adaptave(self, data, key):
		""" the filtering 
		
		input: data => data we want to filtering
		       key => key fo the freq we are filtering 
		       
		globals: self.taps => number of weights
		
		output: the filtered data 
		
		side_effects:
		self.err & self.weights get updated
		self.data_len gets updated (nothing will probs change)
		"""
		
		self.data_len = len(data)
		
		v_data = wave2vol(data)
		v_noise = wave2vol(self.fp[key])
		
		v_data = self.dedup_vol(v_data)
		v_noise = self.dedup_vol(v_noise)
		
		# extend noise so that it is the same length as data
		while len(v_noise) < len(v_data):
			v_noise = np.append(v_noise, v_noise)
		v_noise = v_noise[:len(v_data)]
		
		# calculate nlms
		est_v_noise, self.err, self.weights = self.nlms(v_data, v_noise)
		
		# nlms makes the data offset & slightly shorter
		est_v_noise = np.roll(est_v_noise, self.taps - 1)
		est_v_noise = np.append(est_v_noise, est_v_noise[0:self.taps - 1])
		
		# calculate the estimated data volume
		est_v_data = v_data - est_v_noise
		
		# set negative volume to 0
		est_v_data[est_v_data < 0] = 0
		
		v_data = self.redup_vol(v_data)
		est_v_data = self.redup_vol(est_v_data)
		
		# re-volumise the data
		return data * est_v_data / v_data
	
	def nlms(self, v_data, v_noise):
		""" currently just an alias to adaptfilt """
		return ada.nlms(v_data,
		                v_noise,
		                self.taps,
		                self.mu,
		                eps=self.eps,
		                leak=self.leak,
		                initCoeffs=self.initCoeffs,
		                N=self.n,
		                returnCoeffs=self.returnCoeffs)


# -------------------------------------------------------------------------------------------- #
"""

Probs not going to be persuing this but whatever

try:
	# bit of an esoteric library, so dont error if not found
	import tftb.processing as tftb
except:
	print("tftb not found")
	tftb = False


def noise_gate_PWVD(data, spread=1000):
	
	# something smoothed_pseudo_wigner_ville
	
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
"""
