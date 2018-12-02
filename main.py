import numpy as np
from dft import dft
from cooley_tukey import cooley_tukey


if __name__== "__main__":
	data = np.array([1,2-1j,-1j,-1+2j])
	print(cooley_tukey(data))
	print(list(map(round, dft(data))))
	print(ifft(cooley_tukey(data),cooley_tukey))
	print(list(map(round, ifft(dft(data),dft))))
	data = np.array([[0,1,2+1j,3],[3+2j,-1+4j,8j,0],[-6,-4,2+1j,9-8j],[0,2+1j,2-1j,3]])
	print(fft2d(data,cooley_tukey))
	print(np.fft.fft2(data))
	print(ifft2d(fft2d(data,cooley_tukey), cooley_tukey))
