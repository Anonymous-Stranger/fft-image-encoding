from collections import namedtuple
from functools import partial
from PIL import Image
import numpy as np


CompressedImage = namedtuple(
    'CompressedImage', ['data', 'block_size', 'bounds', 'compressor'])


class ImageCompressor:

    def __init__(self, encoder2d, decoder2d, block_size=8):
        self.encode = encoder2d
        self.decode = decoder2d
        self.block_size = block_size

    def from_array(self, pixels):
        nr, nc = len(pixels), len(pixels) and len(pixels[0])
        assert nr > 0 and nc > 0, "can't compress empty image"

        bsz = 2 ** (max(nr, nc)-1).bit_length()
        bsz = min(bsz, self.block_size)
        padr = (nr+bsz-1) // bsz * bsz - nr
        padc = (nc+bsz-1) // bsz * bsz - nc
        pr0, pc0 = padr // 2, padc // 2
        pr1, pc1 = padr - pr0, padc - pc0

        pixels = np.pad(pixels, ((pr0, pr1), (pc0, pc1), (0, 0)),
                        mode='constant')
        print(pixels.shape)

        bkrows, bkcols = (nr + padr) // bsz, (nc + padc) // bsz
        ncrs = pixels.shape[2]

        # NOTE: Swaps the colors with the rows, so r, c, col -> col, c, r
        pixels = pixels.swapaxes(2, 0).reshape(ncrs, bkcols, bkrows, bsz, bsz)

        block_gen = (((ci, ri, bi), block)
                     for ci, color in enumerate(pixels)
                     for ri, block_row in enumerate(color)
                     for bi, block in enumerate(block_row))

        first = self.encode(next(block_gen)[1])
        shape = (ncrs, bkcols, bkrows, *first.shape)

        data = np.zeros(shape, dtype=first.dtype)
        data[0, 0, 0] = first
        for (ci, ri, bi), block in block_gen:
            data[ci, ri, bi] = self.encode(block)

        return CompressedImage(data, bsz, (pr0, pc0, nr + pr1, nc + pc1), self)

    def to_array(self, compimg):
        bks, shape = compimg.block_size, compimg.data.shape
        ncrs, bkcols, bkrows = shape[0], shape[1], shape[2]
        decoded = np.zeros((ncrs, bkcols, bkrows, bks, bks))
        for ci, color in enumerate(compimg.data):
            for ri, block_row in enumerate(color):
                for bi, block in enumerate(block_row):
                    decoded[ci, ri, bi] = self.decode(block)

        padded = decoded.reshape(ncrs, bkcols*bks, bkrows*bks).swapaxes(0, 2)
        r0, c0, r1, c1 = compimg.bounds
        return padded[r0:r1, c0:c1]


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


if __name__ == "__main__":
    import sys
    from dft import dft
    from cooley_tukey import cooley_tukey

    img = Image.open(sys.argv[1])
    img = img.resize((400, 400 * img.height // img.width))

    ic = ImageCompressor(partial(fft2d, dft),
                         partial(ifft2d, dft))

    # ic = ImageCompressor(np.fft.fft2, np.fft.ifft2)

    pixels = np.array(img)
    print(pixels.shape)

    enc = ic.from_array(pixels)
    print(np.array(enc.data).shape)

    dec = ic.to_array(enc)
    print(dec.shape)

    print(np.mean(np.square(pixels - dec)))
