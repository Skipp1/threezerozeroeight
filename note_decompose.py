#! /usr/bin/env python3

from scipy.io import wavfile
import fft_wrapper as fft
import numpy as np
import os
import h5py
import note_utils as note
import matplotlib.pyplot as plt
import math_fun


class decompose:
	
	def __init__(self, filename):
		
		""" global var declarations """
		
		# tunable paramsdata is now gated
		self.n_cpu = os.cpu_count()
		self.width = 0.5
		self.octaves = (2,6)
		self.noteints = 12
		
		self.filedata = wavfile.read(filename)
		
		# get file information
		self.sample_rate = self.filedata[0]
		self.filedata = self.stereo2mono(self.filedata[1])
		return
	
	def stereo2mono(self, d, channel='a'):
		""" convert stereo audio to mono audio 
		inputs:
		d => numpy array of audio
		channel => how to get to mono
				'l' => only right channel
				'r' => only left channel
				'a' => mean of the channels
				
		return: mono audio 
		side effects: None
		"""
		# mono audio 
		
		if len(d.shape) == 1:
			# just return as we already have a mono signal
			return np.copy(d)
		
		# stereo audio 
		elif len(d.shape) == 2:
			if channel == 'l':
				return d[:,0]
			
			elif channel == 'r':
				return d[:,1]
			
			elif channel == 'a':
				return np.mean(d, axis=1)
			
			else:
				raise Exception("Channel in automid.stereo2mono must be 'a', 'l' or 'r'")
			
		# quadraphonic? (or something else exotic (Hopefully we should never come here))
		else:
			return np.mean(d, axis=1)



	def freq2index(self, fourier_freqs, f):
		""" convert freq to a index on the fourier array
		inputs:
		fourier_freqs => fourier freq conversion
		f => freq we want to find the index of 
		
		return: closest index of desired freq
		side_effects: none
		"""
		return np.argmin(np.abs(fourier_freqs - f))
	
		
	def gauss_select(self, fourier_data, fourier_freqs, octave, noteint):
		""" select a gaussian set of freqs
		input:
		fourier_data => array of fourier transformed data 
		fourier_freqs => what index in fourier_data corresponds to what freq
		octave, noteint => the specific note we want to select
		
		return: selected Fourier data
		side_effects: None
		"""
				
		# get the freq of the notedata is now gated
		f = note.note2freq(octave, noteint)
		
		k= note.freq2note(f)
		
		print(octave, noteint, f, k)
		
		# get the freq of the next note in the series
		fminus1 = note.note2freq(int(np.floor((noteint-1)/ 12) + octave),
		                        int(np.mod(noteint-1, 12)))
		
		mu = self.freq2index(fourier_freqs, f)
		muminus1 = self.freq2index(fourier_freqs, fminus1)
		
		sigma = self.width*(mu - muminus1)
		
		if (mu - muminus1) == 0: 
			# we reached the end of the fourier 
			return np.nan
		
		# create a Gaussian with mu = note,  sigma = 0.5*(note+1 - note)
		g = math_fun.gaussian(self.g_init, mu, sigma)
		
		# multiply to select only the note we are looking at
		ret = fourier_data * g 
		
		return ret 
		

	def decompose(self, filename_out, savetype=0):
		
		# ------ fft transform -------#
		fourier_data = fft.rfft(self.filedata, threads=self.n_cpu, overwrite_input=True)
		
		# convert samples per second to a list of freqs 
		fourier_freqs = np.fft.rfftfreq(len(self.filedata), 1/self.sample_rate)
		# ------ fft transform -------#
		
		fp = h5py.File(filename_out + '.hdf5', 'w', libver='latest')
		
		# save some metadata
		fp.create_dataset('meta',
		                  data=[savetype, self.sample_rate],
		                  dtype=int)
		
		# prevent re-generating gaussian space every time
		self.g_init = np.linspace(0,len(fourier_data), len(fourier_data))
		
		# for each note do:
		for octave in range(*self.octaves):
			for noteint in range(self.noteints):

				# select the note we want to look at
				selected_note = self.gauss_select(fourier_data, fourier_freqs, octave, noteint)
				
				if selected_note is np.nan:
					break
				
				if savetype == 0:
					# inverse fft to convert back to real
					newdata = fft.irfft(selected_note, threads=self.n_cpu, overwrite_input=True)
				elif savetype == 1:
					# just save the fft data
					newdata = selected_note
					
				else:
					raise Exception("No idea how to handle savetype" + savetype)
				
				# save our decomposition
				fp.create_dataset(str(octave) + '-' + note.notenames[noteint], data = newdata, dtype=np.float64)
				
			if selected_note is np.nan:
				print("overrun bounds, safely stopping at ", octave, note.notenames[noteint])
				break
				
		# cleanup
		fp.close()
		return 

