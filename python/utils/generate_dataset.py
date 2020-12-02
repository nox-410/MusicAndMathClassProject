import os
import sys
import argparse
import tempfile
import wave
import random
import numpy as np

# Converting to [n , 2, 441000] numpy array

# used to store generated dataset
stored_data = []
sample_freq = 44100
sample_len = 10 * sample_freq # 10 second

#midi file name starts from
midi_index = 0

def save_npy(test = False):
    data = np.array([i[0] for i in stored_data])
    label = np.array([i[1] for i in stored_data])
    if test:
        np.save(os.path.join(args.path, "test_data.npy"), data)
        np.save(os.path.join(args.path, "test_label.npy"), label)
    else:
        np.save(os.path.join(args.path, "data.npy"), data)
        np.save(os.path.join(args.path, "label.npy"), label)

# try to find the frequency file for timidity
def get_frequency_file():
    path = os.path.dirname(__file__)
    path = os.path.join(path, "..")
    path = os.path.join(path, "..")
    path = os.path.join(path, "freq_table")
    path = os.path.abspath(path)
    freq_files = [
        os.path.join(path, "equal.txt"),
        os.path.join(path, "pure.txt"),
        os.path.join(path, "pythagorean.txt")
    ]
    return freq_files

def read_wav(fname):
    with wave.open(fname) as f:
        params = f.getparams()
        framesra, frameswav= params[2],params[3]
        datawav = f.readframes(frameswav)

    data = np.frombuffer(datawav,dtype = np.short)
    data = data.reshape(-1, 2).T
    return data

def convert_and_save(cmd, wavfile, label):
    os.system(cmd)
    data = read_wav(wavfile)
    data = data[:, 5 * sample_freq : -5 * sample_freq] # trim start and end
    start = 0
    while start + sample_len < data.shape[1]:
        wav_slice = data[:,start : start + sample_len]
        start = start + sample_len
        if len(stored_data) < int(args.n):
            stored_data.append((wav_slice, label))
    os.remove(wavfile)

def main():
    global midi_index
    freqs = get_frequency_file()
    while True:
        midi = "{}.mid".format(midi_index)
        midi_index += 1
        midi = os.path.abspath(os.path.join(args.midipath, midi))
        for i, freq in enumerate(freqs):
            _ ,wavfile = tempfile.mkstemp() # tempfile to store .wav
            cmd = ["timidity", "-Ow", "-o", wavfile]  + ["-Z", freq, midi]
            cmd = cmd + [">>", "/dev/null"] # quieter
            cmd = " ".join(cmd)
            print(cmd)
            convert_and_save(cmd, wavfile, i)
        if len(stored_data) >= int(args.n):
            break

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--midipath", required=True, help="Path to midi directory")
    parser.add_argument("--path", "-p", required=True, help="Path to place the dataset")
    parser.add_argument("-n", default=1000, help="total size of the dataset")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        os.mkdir(args.path)
    main()
    save_npy(test=False)
    # also generate test dataset
    print("--- Generating test dataset ---")
    stored_data = []
    main()
    save_npy(test=True)
    print("--- Done ---")
