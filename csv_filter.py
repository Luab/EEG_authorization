import csv
import ast
import scipy.signal as sig
import numpy
from scipy.io.wavfile import write

with open('Maksudov_bulat_green_circle.csv') as csvfile:
	reader = csv.reader(csvfile)
	f = open("row3.csv","w")
	y = numpy.empty(1)
	for row in reader:
		numpy.append(y,row[2])
	x = sig.resample(y,y.size*5*44100)
	scaled = numpy.int16(x/numpy.max(numpy.abs(x))*32767)
	write("test.wav",44100,scaled)			
