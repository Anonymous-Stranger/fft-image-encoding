from PIL import Image
import numpy as np
import sys


from img_dft import img_encode, img_decode
from dft import dft
from cooley_tukey import cooley_tukey


if __name__ == "__main__":
    algorithm = cooley_tukey
    if len(sys.argv) > 1:
        img = Image.open(sys.argv[1])
        encoded = img_encode(img, algorithm)
        for i in range(8, 16):
            for j in range(8, 16):
                encoded[i, j] = 0
        # print(encoded)
        decoded = Image.fromarray(img_decode(encoded, algorithm))
        decoded.show()
        # print(np.array(img)[:, :, 1] - decoded[:, :, 1])
