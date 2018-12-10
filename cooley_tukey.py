import numpy as np


def cooley_tukey(data):
    data = data.astype(np.complex64)
    if(data.size == 1):
        return data
    even = cooley_tukey(data[::2])
    odd = cooley_tukey(data[1::2])
    multiplier = np.exp(np.pi*-2j/data.size)
    k = 1
    for i in range(0, odd.size):
        odd[i] *= k
        k *= multiplier
    return np.concatenate((even+odd, even-odd))
