#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import note_recompose as nre
import noise_gate as ngate
import note_utils as note
import sys
import math_fun


def main():
	
	print("decomposing data")
	nde_class = nde.decompose('test.wav')
	nde_class.octaves = (2, 15)
	nde_class.decompose('ns~test')
	
	print("decomposing bg")
	#ngs_class = ngate.noise_gate_sigma('bg_fan.wav', nde_class.octaves)
	nga_class = ngate.noise_gate_adaptave('bg_fan.wav', nde_class.octaves)
	nga_class.spread = 1000
	nga_class.mu = 0.1
	
	fp_in = h5py.File('ns~test.hdf5', 'r', libver='latest')
	fp_out = h5py.File('ns~test2.hdf5', 'w', libver='latest')
	
	# copy over the meta key
	fp_out.create_dataset('meta', data=fp_in['meta'], dtype=int)
	
	for i, key in enumerate(fp_in.keys()):
		if key == 'meta':
			continue
		
		# print some stuff
		d = np.copy(fp_in[key])
		freq = note.note2freq(int(key.split('-')[0]), key.split('-')[1])
		print("filtering:  %s \t %.2f Hz " % (key, freq))
		
		#d = ngs_class.noise_gate_sigma(d, key, 5, spread=1000)
		#d = ngate.noise_gate_PWVD(d, spread=1000)
		d = nga_class.noise_gate_adaptave(d, key)
		
		fp_out.create_dataset(key, data=d, dtype=d.dtype)
	
	print("recomposing")
	nre.recompose('ns~test2', 'out.wav')
	
	fp_in.close()
	fp_out.close()
	
	return


if __name__ == "__main__":
	main()
