import numpy as np
from dft import dft
from cooley_tukey import cooley_tukey

def ifft(data, fftAlgo):
	return fftAlgo(np.concatenate(([data[0]],np.flip(data[1:],0))))/data.size

if __name__== "__main__":
	data = np.array([1,2-1j,-1j,-1+2j])
	print(cooley_tukey(data))
	print(list(map(round, dft(data))))
	print(ifft(cooley_tukey(data),cooley_tukey))
	print(list(map(round, ifft(dft(data),dft))))
