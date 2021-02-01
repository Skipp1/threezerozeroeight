#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import note_recompose as nre
import noisegate as ngate
import note_utils as note
import math_fun

def main():
	
	print("decomposing")
	nde_class = nde.decompose('50Hz+1250Hz.wav')
	nde_class.octaves = (2,15)
	nde_class.decompose('ns~test')
	
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
	
		d = ngate.noisegate(0.1*math_fun.pink_power(freq, alpha=2), d, spread=1000)

		fp_out.create_dataset(key, data=d, dtype=d.dtype)

	print("recomposing")
	nre.recompose('ns~test2', '50Hz+1250Hz-clean')
	
	fp.close()
	fp_out.close()
	
	return	
	
	
if __name__ == "__main__":
	main()
