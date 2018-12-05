import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

from dft import dft
from cooley_tukey import cooley_tukey

if __name__=="__main__":
	sizes = [2,4,8,16,32,64,128,256,512]
	dftTimeAvg = [0]*len(sizes)
	fftTimeAvg = [0]*len(sizes)
	npTimeAvg = [0]*len(sizes)
	for i in range(0,len(sizes)):
		for j in range(0,10):
			arr = 100*((np.random.rand(sizes[i])-.5) + 1j*(np.random.rand(sizes[i])-.5))
			start = timer()
			dft(arr)
			end = timer()
			dftTimeAvg[i] += end-start
			start = timer()
			cooley_tukey(arr)
			end = timer()
			fftTimeAvg[i] += end-start
			start = timer()
			np.fft.fft(arr)
			end = timer()
			npTimeAvg[i] += end-start
		dftTimeAvg[i]/=10
		fftTimeAvg[i]/=10
		npTimeAvg[i]/=10

	plt.subplot(1,2,1)
	plt.title("Time elapsed vs Input size")
	plt.xlabel("Input size")
	plt.ylabel("Seconds")
	plt.plot(sizes,dftTimeAvg,'ro-',label='dft')
	plt.plot(sizes,fftTimeAvg,"gs-",label='fft')
	plt.plot(sizes,npTimeAvg,"b^-",label='numpy')
	plt.legend(loc='upper left')
	plt.subplot(1,2,2)
	plt.title("Time elapsed vs Input size (log scaled)")
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel("Input size (log scale)")
	plt.ylabel("Seconds (log scale)")
	plt.plot(sizes,dftTimeAvg,'ro-',label='dft')
	plt.plot(sizes,fftTimeAvg,"gs-",label='fft')
	plt.plot(sizes,npTimeAvg,"b^-",label='numpy')
	plt.legend(loc='upper left')
	plt.show()
