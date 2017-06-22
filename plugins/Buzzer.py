import plugin_interface as plugintypes
from scipy.io.wavfile import write
import numpy
class PluginBuzz(plugintypes.IPluginExtended):
	def activate(self):
		self.arr = [0]
		#self.arr[0] = 0
		self.i = 1
		print("Buzz activated")


	# called with each new sample
	def __call__(self, sample):
		if sample:
			#print(sample.channel_data[1])
			#print(self.arr[-1])
			#print((self.arr[-1]+sample.channel_data[1])/2)
			self.arr.append((self.arr[-1]+sample.channel_data[1])/2)
			self.arr.append((self.arr[-2]+self.arr[-1])/2)
			self.arr.append(sample.channel_data[1])
			#print(str(len(self.arr)))
			if len(self.arr) > self.i*44100:
				print(str(self.i))
				x = numpy.asarray(self.arr)
				scaled = numpy.int16(x/numpy.max(numpy.abs(x))*32767)
				write("test.wav",44100,scaled)
				self.i = self.i+1
		# DEBBUGING
		# try:
		#     sample_string.decode('ascii')
		# except UnicodeDecodeError:
		#     print("Not a ascii-encoded unicode string")
		# else:
		#     print(sample_string)
		
