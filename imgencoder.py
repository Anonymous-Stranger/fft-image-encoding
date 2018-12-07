from collections import namedtuple
import numpy as np


EncodedImage = namedtuple(
    'CompressedImage', ['data', 'block_size', 'bounds', 'compressor'])


class ImageEncoder:

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

        return EncodedImage(data, bsz, (pr0, pc0, nr + pr1, nc + pc1), self)

    def to_array(self, encimg):
        bks, shape = encimg.block_size, encimg.data.shape
        ncrs, bkcols, bkrows = shape[0], shape[1], shape[2]
        decoded = np.zeros((ncrs, bkcols, bkrows, bks, bks))
        for ci, color in enumerate(encimg.data):
            for ri, block_row in enumerate(color):
                for bi, block in enumerate(block_row):
                    decoded[ci, ri, bi] = np.real(self.decode(block))

        padded = decoded.reshape(ncrs, bkcols*bks, bkrows*bks).swapaxes(0, 2)
        r0, c0, r1, c1 = encimg.bounds
        return padded[r0:r1, c0:c1]
