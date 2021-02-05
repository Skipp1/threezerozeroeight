#! /usr/bin/env python3

import numpy as np

# NOTE: octaves start and end on C, not A

# Dict of notenames 
notenames = {0:  'A', 1:  'A#',
             2:  'B',	 
             3:  'C', 4:  'C#',
             5:  'D', 6:  'D#',
             7:  'E', 
             8:  'F', 9:  'F#',
             10: 'G', 11: 'G#'}


def freq2note(f):
	""" converts freq to the nearest note 
	
	input: f => freq
	
	return: (octave (int), note (str))
	side effects: None
	"""
	
	n = 12 * np.log2(f / 440)
	n =  np.round(n)
	
	octave = np.floor(n/12) + 4
	note = np.mod(n, 12)
	
	return int(octave), notenames[note]

def note2freq(octave, note):
	""" converts freq to the nearest note 
	
	input: octave (int)
	       note (int) or (str)
	
	return: frequency
	side effects: None
	"""
	
	
	if type(note) != int:
		# convert str to int
		#if np.floor((list(notenames.values()).index(note)+3) /12) == 1:
			#octave += 1
		note = np.mod(list(notenames.values()).index(note), 12)

		
	n = note + 12*int(octave - 4)
	return 440 * (2 **(n/12))

