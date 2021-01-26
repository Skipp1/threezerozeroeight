#! /usr/bin/env python3

from scipy.io import wavfile
import pyfftw # faster fft than numpy 
import numpy as np
import note_utils as note
import os
import math_fun


class decompose(self):
	
	# prevent re-generating gaussian space every time
	g_init = np.linspace(0,len(fourier_data), len(fourier_data))
	
	# tunable params
	n_cpu = os.cpu_count()
	filedata = None
	width = 0.5
	
	def __init__(filename):
		
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
		if len(d.shape == 1):
			# just return as we already have a mono signal
			return d
		
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
				
		# get the freq of the note
		f = note.note2freq(octave, noteint)
		
		# get the freq of the next note in the series
		fplus1 = note.note2freq(int(np.floor((noteint+1)/ 12) + octave), int(np.mod(noteint+1, 12)))
		
		mu = freq2index(fourier_freqs, f)
		sigma = self.width*(freq2index(fourier_freqs, fplus1) - mu)
		
		# create a Gaussian with mu = note,  sigma = 0.5*(note+1 - note)
		g = math_fun.gaussian(self.g_init, mu, sigma)
		
		# multiply to select only the note we are looking at
		ret = fourier_data * g 
		
		return ret 
		


	def decompose(self):
		
		# ------ fft transform -------#
		fourier_data = pyfftw.interfaces.numpy_fft.rfft(self.filedata, threads=self.n_cpu, overwrite_input=True)
		
		# convert samples per second to a list of freqs 
		fourier_freqs = np.fft.rfftfreq(len(self.filedata), 1/self.sample_rate)
		# ------ fft transform -------#
		
		out_filenames = []
		
		# for each note do:
		for octave in range(2,6):
			for noteint in range(12):

				# select the note we want to look at
				selected_note = gauss_select(fourier_data, fourier_freqs, octave, noteint)
				
				# inverse fft
				newdata = pyfftw.interfaces.numpy_fft.irfft(selected_note, threads=self.n_cpu, overwrite_input=True)
				
				# how are we going to transform real + imag to only real?
				savedata = np.imag(newdata) + np.real(newdata)
				
				# save as a new wav file
				save_filename = os.path.normpath("decomposition/" + str(octave) + note.notenames[noteint] + ".wav")
				wavfile.write(save_filename, sample_rate, savedata)
				
				out_filenames.append(save_filename)
						
		return out_filenames
