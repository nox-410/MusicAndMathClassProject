# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 10:13:28 2020

@author: WTY
"""
import os
import librosa as lbr
import numpy as np

SAMPLE_RATE = 8000
OVERRIDE_PATH = './dataset'
SAVE_PATH = './wavset'

def wav_to_npy(in_path, out_path):
    print('%s -> %s' % (in_path, out_path))
    waveform, sample_rate = lbr.load(in_path)
    waveform = lbr.resample(waveform, orig_sr = sample_rate, target_sr = SAMPLE_RATE)
    np.save(out_path, waveform)    

if __name__ == '__main__':
    i = 1
    for d in os.listdir(OVERRIDE_PATH):
        p = os.path.join(OVERRIDE_PATH, d)
        s = os.path.join(SAVE_PATH, d)
        if (not os.path.isdir(p)) or d.startswith('.'):
            continue
        for w in os.listdir(p):
            if w.startswith('.') or not w.endswith('.wav'):
                continue
            n = os.path.splitext(w)[0] + '.npy'
            wav_path = os.path.join(p, w)
            npy_path = os.path.join(s, n)
            print('[%5d]' % i, end=' ')
            wav_to_npy(wav_path, npy_path)
            i = i + 1