#!/usr/bin/env python3
from PIL import Image
import csv
import numpy as np
import os

from imgencoder import ImageEncoder
from fft2d import encode_and_compress, uncompress_and_decode
from dft import dft
from cooley_tukey import cooley_tukey


img_dir = 'img/'
compression_rates = [0.5, 0.75, .8, .9, .95, 1]
algorithms = [
    ('numpy fft', np.fft.fft),
    ('dft', dft),
    ('cooley_tukey', cooley_tukey)
]
img_h = 400
block_size = 8


def test_accuracy(alg, comp_rate, img_array):
    max_term = np.int8(block_size*block_size*comp_rate)
    encoder = ImageEncoder(encode_and_compress(alg, max_term),
                           uncompress_and_decode(alg, block_size),
                           block_size=block_size)
    enc = encoder.from_array(img_array)
    dec = encoder.to_array(enc)
    return enc.data.nbytes, rms_error(img_array, dec)


def rms_error(orig, decoded):
    return np.sqrt(np.mean(np.square(orig - decoded)))


if __name__ == "__main__":
    idencoder = ImageEncoder(lambda x: x.astype(np.complex64),
                             lambda x: x, block_size=block_size)

    with open('test_results/accuracy.csv', 'w') as outfile:
        out = csv.writer(outfile)
        out.writerow([
            'image', 'image_height', 'algorithm', 'block_size',
            'compression_rate', 'encoded size (bytes)', 'rms error'
        ])

        for iname in os.listdir(img_dir):
            img = Image.open(os.path.join(img_dir, iname))
            pixels = np.array(img.resize((img_h*img.width//img.height, img_h)))
            nbytes = idencoder.from_array(pixels).data.nbytes

            out.writerow([
                iname, img_h, 'identity transform', block_size, 1, nbytes, 0
            ])
            print("{} ({} KB)".format(iname, nbytes // 1000))

            for aname, alg in algorithms:
                for comp_rate in compression_rates:
                    nbytes, err = test_accuracy(alg, comp_rate, pixels)
                    out.writerow([
                        iname, img_h, aname, block_size, comp_rate, nbytes, err
                    ])
                    print(
                        "    {} (comp. rate: {}): {} KB, rms error: {}".format(
                            aname, comp_rate, nbytes // 1000, err
                        )
                    )
