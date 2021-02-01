#! /usr/bin/env python3
"""
Because some people in windows are having trouble installing pyfftw, 
fall back to numpy (or another fft program if you want to use that here)
"""

# select what fft we are going to use
try:
	import pyfftw
	fft_ver = 'pyfftw'
except:
	import numpy
	fft_ver = 'numpy'
	
	
	
def fft_interface_select():
	""" return the interface class for the fft """
	
	if fft_ver == 'pyfftw':
		return  pyfftw.interfaces.numpy_fft
	
	elif fft_ver == 'numpy':
		return numpy.fft
	
	else:
		raise Exception("Dont know this fft type")
	
	
def fft_interface_format(fft_method, data, **kargs):
	""" run the fft /ifft over data using kargs """
	if fft_ver == 'pyfftw':
		#pyfftw accepts a superset of numpy
		return fft_method(data, **kargs)
	
	elif fft_ver == 'numpy':
		# select only the args that are compatible with numpy 
		new_kargs = {}
		for i in ('n', 'a', 'axis', 'norm'):
			try:
				new_kargs[i] = kargs[i]
			except KeyError:
				# the arg wasnt passed in the first place
				pass
			
		return fft_method(data, **new_kargs)
		
	
def fft(data, **kargs):
	""" fft wrapper """
	interface = fft_interface_select() 
	return fft_interface_format(interface.fft, data, **kargs)
		
def ifft(data, **kargs):
	""" ifft wrapper """
	interface = fft_interface_select() 
	return fft_interface_format(interface.ifft, data, **kargs)
	
def rfft(data, **kargs):
	""" rfft wrapper """
	interface = fft_interface_select() 
	return fft_interface_format(interface.rfft, data, **kargs)
		
def irfft(data, **kargs):
	""" irfft wrapper """
	interface = fft_interface_select() 
	return fft_interface_format(interface.irfft, data, **kargs)

