import numpy as np
from cooley_tukey import cooley_tukey

if __name__== "__main__":
	print(cooley_tukey(np.array([1,2-1j,-1j,-1+2j])))
