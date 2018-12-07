#!/usr/bin/env python3
from collections import namedtuple
from PIL import Image
from time import process_time
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


AccuracyResults = namedtuple(
    'AccuracyResults', ['nbytes', 'error', 'enctime', 'dectime'])


def test_accuracy(alg, comp_rate, img_array):
    max_term = np.int8(block_size*block_size*comp_rate)
    encoder = ImageEncoder(encode_and_compress(alg, max_term),
                           uncompress_and_decode(alg, block_size),
                           block_size=block_size)
    return test_encoder_acc(encoder, img_array)


def test_encoder_acc(encoder, img_array):
    start_time = process_time()
    enc = encoder.from_array(img_array)
    enc_time = process_time()
    dec = encoder.to_array(enc)
    dec_time = process_time()
    return AccuracyResults(enc.data.nbytes, rms_error(img_array, dec),
                           enc_time - start_time, dec_time - enc_time)


def rms_error(orig, decoded):
    return np.sqrt(np.mean(np.square(orig - decoded)))


if __name__ == "__main__":

    id_encoder = ImageEncoder(lambda x: x.astype(np.complex_),
                              lambda x: x, block_size=block_size)

    with open('test_results/accuracy.csv', 'w') as outfile:
        out = csv.writer(outfile)
        out.writerow([
            'image', 'image_height', 'algorithm', 'block_size',
            'compression_rate', 'encoded size (bytes)', 'rms error',
            'encoding time (s)', 'decoding time (s)'
        ])

        for iname in os.listdir(img_dir):
            img = Image.open(os.path.join(img_dir, iname))
            pixels = np.array(img.resize((img_h*img.width//img.height, img_h)))

            res = test_encoder_acc(id_encoder, pixels)

            out.writerow([
                iname, img_h, 'identity transform', block_size, 1,
                res.nbytes, res.error, res.enctime, res.dectime
            ])
            print("{} ({} KB)".format(iname, res.nbytes // 1000, res.error))

            # for aname, alg in algorithms:
            #     for comp_rate in compression_rates:
            #         res = test_accuracy(alg, comp_rate, pixels)
            #         out.writerow([
            #             iname, img_h, aname, block_size, comp_rate,
            #             res.nbytes, res.error, res.enctime, res.dectime
            #         ])
            #         print("[{:4.2g} s] {} (comp. rate: {}): "
            #               "{} KB, rms error: {}".format(
            #                 res.enctime + res.dectime, aname, comp_rate,
            #                 res.nbytes // 1000, res.error))
