#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import noisegate as ngate

def main():
	nde_class = nde.decompose('test.wav')
	nde_class.decompose('test')
	
	#ng_class = ngate.noisegate()
	
	fp = h5py.File('test.hdf5', 'r', libver='latest')
	fp_out = h5py.File('test2.hdf5', 'w', libver='latest')
	
	# recreate the meta key
	fp_out.create_dataset('meta', data=fp['meta'], dtype=int)
	
	for key in list(fp.keys()):
		
		# skip over the meta key 
		if key == 'meta':
			continue
		
		d = fp[key]
		
		d = ngate.noisegate(100, d)
		
		print(key)
		
		fp_out.create_dataset(key, data=d, dtype=d.dtype)
		
	nde_class.recompose('test2')
	return	
	
	
if __name__ == "__main__":
	main()
