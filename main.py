#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import note_recompose as nre
import noise_gate as ngate
import note_utils as note
import sys
import math_fun
sudo pip install adaptfilt
import adaptfilt as adf






def main():
	
	print("decomposing data")
	nde_class = nde.decompose('test.wav')
	nde_class.octaves = (2, 15)
	nde_class.decompose('ns~test')
	
	print("decomposing bg")
	ng_class = ngate.noise_gate_sigma('bg_fan.wav', nde_class.octaves)
	
	fp = h5py.File('ns~test.hdf5', 'r', libver='latest')
	fp_out = h5py.File('ns~test2.hdf5', 'w', libver='latest')
	
	# copy over the meta key
	fp_out.create_dataset('meta', data=fp['meta'], dtype=int)
	
	for i, key in enumerate(fp.keys()):
		if key == 'meta':
			continue
		
		d = np.copy(fp[key])
		freq = note.note2freq(int(key.split('-')[0]), key.split('-')[1])
		print("noise gating: %s \t %.2f Hz " % (key, freq))
		
		d = ngs_class.noise_gate_sigma(d, key, 2, spread=1000)
		#d = ngs_class.noise_gate_PWVD(d, spread=1000)
		
		fp_out.create_dataset(key, data=d, dtype=d.dtype)
	
	print("recomposing")
	nre.recompose('ns~test2', 'out.wav')
	
	y, e, w = adf.nlms(u, d, M, step, eps=0.001, leak=0, initCoeffs=None, N=None, returnCoeffs=False)
	u=nde_class # decomposed wave
	d=          # recomposed wave or a guess of the desired wave 
	M=100
	step=1
	
	plt.figure()
        plt.title("Noise cancellation")
        plt.plot(u, label="initial signal")
        plt.plot(d, label="desired siganl")
	plt.plot(y, label="output signal")
	
	
	return 
	
	fp.close()
	fp_out.close()
	
	return


if __name__ == "__main__":
	main()
