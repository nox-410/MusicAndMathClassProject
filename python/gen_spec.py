import os
import numpy as np
import torchaudio
from torchaudio.compliance import kaldi

N_FFT = 4096
SAMPLE_RATE = 8000
OVERRIDE_PATH = './dataset'

transform = torchaudio.transforms.Spectrogram(N_FFT)


def wav_to_spec(in_path, out_path):
    print('%s -> %s' % (wav_path, npy_path))
    waveform, sample_rate = torchaudio.load(in_path)
    waveform = waveform[:1]

    shape = waveform.size()
    waveform = waveform.view(-1, shape[-1])
    waveform = kaldi.resample_waveform(
        waveform,
        orig_freq=sample_rate,
        new_freq=SAMPLE_RATE,
    )
    waveform = waveform.view(shape[:-1] + waveform.shape[-1:])

    specgram = transform(waveform)  # (channels, bins, frames)
    specgram = specgram.numpy()

    np.save(out_path, specgram)
    os.remove(in_path)


if __name__ == '__main__':
    i = 1
    for d in os.listdir(OVERRIDE_PATH):
        p = os.path.join(OVERRIDE_PATH, d)
        if (not os.path.isdir(p)) or d.startswith('.'):
            continue
        for w in os.listdir(p):
            if w.startswith('.') or not w.endswith('.wav'):
                continue
            n = os.path.splitext(w)[0] + '.npy'
            wav_path = os.path.join(p, w)
            npy_path = os.path.join(p, n)
            print('[%5d]' % i, end=' ')
            wav_to_spec(wav_path, npy_path)
            i = i + 1
