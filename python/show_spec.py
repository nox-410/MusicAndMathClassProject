import sys
import numpy as np
from PIL import Image


def spec_to_png(in_path, out_path):
    specgram = np.load(in_path)  # (channels, bins, frames)
    specgram = specgram[0]
    specgram = np.log2(specgram)
    specgram = specgram.sum(1)[:, np.newaxis]
    specgram = np.repeat(specgram, 128, axis=1)
    smax, smin = np.max(specgram), np.min(specgram)
    specgram = (specgram - smin) / (smax - smin)
    specgram = (specgram * 256).astype(np.uint8)
    specgram = np.flipud(specgram)
    Image.fromarray(specgram).save(out_path)


if __name__ == '__main__':
    spec_to_png(sys.argv[1], sys.argv[2])
