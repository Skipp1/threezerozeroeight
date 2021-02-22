#! /usr/bin/env python3

import numpy as np
import fft_wrapper as fft
import note_decompose as nde
import matplotlib.pyplot as plt


def savefig_fix(plt, filename, dpi=100):
	""" matplot lib currently has a bug in 3.3.0 where it produces a Unicode minus 
	instead of - this causes pdflatex to freak out and not work. thus, a little
	find and replace should fix that """
	
	plt.savefig(filename, bbox_inches='tight', dpi=dpi)
	
	if filename.split('.')[-1] == 'pgf':
		fp = open(filename, 'rb')
		pgftext = fp.read()
		fp.close()
		
		pgftext = pgftext.replace(bytes(u"\u2212", 'utf-8'), bytes('-', 'utf-8'))
		
		fp = open(filename, 'wb')
		fp.write(pgftext)
		fp.close()
	plt.clf()
	return


def main():
	""" plot some data """
	
	nde_class = nde.decompose("out.wav")
	fourier = False
	
	if fourier:
		y = fft.rfft(nde_class.filedata, threads=nde_class.n_cpu, overwrite_input=True)
		x = np.fft.rfftfreq(len(nde_class.filedata), 1 / nde_class.sample_rate)
	else:
		y = nde_class.filedata
		x = np.arange(len(nde_class.filedata)) / nde_class.sample_rate
	
	a = 3 / 4  # aspect ratio
	s = 5  # scale
	
	plt.figure(figsize=(204.12683 * s / 72.27, (204.12683 * s / 72.27) * a))
	plt.rcParams.update({
	    'font.family': 'serif',  # use a serif font
	    'font.serif': 'CMR10',  # computer modern
	    'text.usetex': True,  # use inline math for ticks
	    'pgf.rcfonts': False,  # don't setup fonts from rc parameters
	    'pgf.texsystem': 'xelatex',  # pdflatex would go here but matplotlib has a bug
	    'font.size': 20
	})
	
	plt.plot(x, y)
	savefig_fix(plt, "out.pgf")
	return


if __name__ == "__main__":
	main()
