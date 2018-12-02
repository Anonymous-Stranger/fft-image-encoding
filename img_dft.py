from functools import partial
from PIL import Image
import numpy as np


def ifft(fft_algo, data):
    return fft_algo(np.concatenate(
        ([data[0]], np.flip(data[1:], 0)))) / data.size


def fft2d(fft_algo, data):
    return np.transpose(np.array([fft_algo(x) for x in np.transpose(
        np.array([fft_algo(x) for x in data]))]))


def ifft2d(fft_algo, data):
    return np.transpose(np.array([ifft(fft_algo, x) for x in np.transpose(
        np.array([ifft(fft_algo, x) for x in data]))]))


def img_fft(img, algorithm):
    return np.stack(list(algorithm(img[:, :, c]) for c in range(4)), -1)


def img_encode(img, algorithm):
    return img_fft(np.asarray(img, dtype=np.complex_),
                   partial(fft2d, algorithm))


def img_decode(img, algorithm):
    return np.asarray(img_fft(img, partial(ifft2d, algorithm)), dtype=np.uint8)
