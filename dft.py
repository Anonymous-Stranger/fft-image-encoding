import numpy as np


def dft(pts, k_max=None):
    k_max = k_max or len(pts)
    return np.array([dftk(pts, k) for k in range(k_max)])


def dftk(pts, k):
    coeff = -2*np.pi*k/len(pts)
    return sum((x * np.exp(complex(0, coeff*n)) for n, x in enumerate(pts)))
