# Split the .wav file specified in command line into 30-second segments;
# the last segment would be discarded if it is shorter than 30 seconds.
import os
import sys
from pydub import AudioSegment
import argparse


def split_wav(wave_path, out_path=None, duration=30*1000):
    dirname, basename = os.path.split(wave_path)
    name, ext = os.path.splitext(basename)
    if out_path is None:
        out_path = dirname

    old_audio = AudioSegment.from_wav(wave_path)

    num_chunks = len(old_audio) // duration
    for i in range(num_chunks):
        t1 = i * duration
        t2 = (i + 1) * duration
        split_audio = old_audio[t1:t2]
        split_audio.export(os.path.join(
            out_path, name + "_{}.wav".format(i)), format="wav")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument(
        "-o", "--output", help="Directory to store splits obtained.")
    args = parser.parse_args()

    split_wav(args.file, args.output)
