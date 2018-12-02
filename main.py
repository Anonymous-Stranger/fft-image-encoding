import numpy as np
from dft import dft
from cooley_tukey import cooley_tukey

if __name__== "__main__":
	data = np.array([1,2-1j,-1j,-1+2j])
	print(cooley_tukey(data))
	print(map(round, dft(data))
