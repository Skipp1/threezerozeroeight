#! /usr/bin/env python3

import numpy as np
import h5py
import note_decompose as nde
import noisegate as ngate

def main():
	
	print("decomposing")
	nde_class = nde.decompose('test.wav')
	nde_class.octaves = (2,10)
	nde_class.decompose('test')
	
	fp = h5py.File('test.hdf5', 'r', libver='latest')
	fp_out = h5py.File('test2.hdf5', 'w', libver='latest')
	
	# copy over the meta key
	fp_out.create_dataset('meta', data=fp['meta'], dtype=int)
	
	for i, key in enumerate(fp.keys()):
		if key == 'meta':
			continue
		
		print("noise gating:", key)
		d = np.copy(fp[key])
	
		d = ngate.noisegate(0.02, d, spread=1000)

		
		fp_out.create_dataset(key, data=d, dtype=d.dtype)

	print("recomposing")
	nde.recompose('test2', 'out')
	
	fp.close()
	fp_out.close()
	
	return	
	
	
if __name__ == "__main__":
	main()
