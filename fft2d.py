import numpy as np


def encode_and_compress(fft_algo, max_term):
    def encoder(img):
        enc = fft2d(fft_algo, img)
        bound, mx = enc.shape[0], min(enc.size, max_term)
        return np.array([enc[x, y] for x, y in zigzag(bound, mx)])
    return encoder


def uncompress_and_decode(fft_algo, block_size):
    def decoder(data):
        full = np.zeros((block_size, block_size), dtype=np.complex64)
        for (x, y), v in zip(zigzag(block_size, len(data)), data):
            full[x, y] = v
        return ifft2d(fft_algo, full)
    return decoder


def fft2d(fft_algo, data):
    return np.transpose(np.array([fft_algo(x) for x in np.transpose(
        np.array([fft_algo(x) for x in data]))]))


def ifft2d(fft_algo, data):
    return np.transpose(np.array([ifft(fft_algo, x) for x in np.transpose(
        np.array([ifft(fft_algo, x) for x in data]))]))


def ifft(fft_algo, data):
    return fft_algo(np.concatenate(
        ([data[0]], np.flip(data[1:], 0)))) / data.size


def zigzag(bound, max_term):
    bound -= 1
    row, col, dir = 0, 0, -1
    for _ in range(max_term):
        yield row, col
        if (col == 0 and dir < 0) or (col == bound and dir > 0):
            row += 1
            dir = -dir
        elif row == 0 and dir > 0 or (row == bound and dir < 0):
            col += 1
            dir = -dir
        else:
            row -= dir
            col += dir
