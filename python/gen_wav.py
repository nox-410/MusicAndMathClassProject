import os
import wave
import random
import tempfile
from subprocess import Popen, DEVNULL
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import List, Text

MIDI_PATH = './clean_midi'
SAVE_PATH = './dataset'
FREQ_TABLE_PATH = './freq_table'
SAMPLES = 100 * 3  # must be multiple of 3
PARALLEL = 8


def find_all_midi(path) -> List[Text]:
    midi_path = []
    for d in os.listdir(path):
        p = os.path.join(path, d)
        if not os.path.isdir(p):
            continue
        for m in os.listdir(p):
            if m.startswith('.') or not m.endswith('.mid'):
                continue
            midi_path.append(os.path.join(p, m))
    midi_path.sort()
    return midi_path


def midi_to_wav(in_path, out_path, freq_table=None):
    cmd = ['timidity', '-Ow', '-o', out_path]
    if freq_table:
        cmd.extend(['-Z', freq_table])
    cmd.append(in_path)
    try:
        p = Popen(cmd, stdout=DEVNULL, stderr=DEVNULL)
        assert p.wait() == 0
    except Exception:
        def repr_safe(s):
            if set(s) & set('"&$ ,'):
                return repr(s)
            return s
        cmdline = ' '.join(repr_safe(i) for i in cmd)
        raise RuntimeError(cmdline) from None


def cut_wav(in_path, out_path, begin_sec=None, dura_sec=30.):
    with wave.open(in_path, 'rb') as f:
        nchannels, sampwidth, framerate, nframes, *_ = f.getparams()
        assert sampwidth == 2, (
            "doesn't support sample width other that 16-bit, current: %d" %
            sampwidth * 8)

        read_frame = int(dura_sec * framerate)
        if begin_sec is None:
            assert read_frame <= nframes, 'duration out of range'
            begin_frame = random.randint(0, nframes - read_frame)
        else:
            begin_frame = int(begin_sec * framerate)
            assert begin_frame + read_frame <= nframes, 'read out of range'

        # each frame is stored as (L, R) if it's stereo
        f.readframes(begin_frame)
        data = f.readframes(read_frame)

    with wave.open(out_path, 'wb') as f:
        f.setnchannels(nchannels)
        f.setsampwidth(sampwidth)
        f.setframerate(framerate)
        f.writeframes(data)


completed = 0
completed_lock = threading.Lock()


def task(i, temper):
    global completed

    def progress():
        return '[%5d/%5d]' % (completed, SAMPLES)
    noret = {'end': '', 'flush': True}

    in_path = midi_path[i]
    out_dir = os.path.join(SAVE_PATH, temper)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '%05d.wav' % i)
    _, tmp_path = tempfile.mkstemp('.wav')
    print('%s > %s\n%s\r' % (in_path, out_path, progress()), **noret)

    try:
        freq_table = os.path.join(FREQ_TABLE_PATH, '%s.txt' % temper)
        midi_to_wav(in_path, tmp_path, freq_table)
        cut_wav(tmp_path, out_path)
    except Exception as e:
        print(e)
        raise e from None
    finally:
        os.remove(tmp_path)

    with completed_lock:
        completed += 1
        print('%s\r' % progress(), **noret)


if __name__ == '__main__':
    midi_path = find_all_midi(MIDI_PATH)
    print('dataset size:', len(midi_path))
    random.seed(0)
    sample = random.sample(range(len(midi_path)), SAMPLES)
    print('samples size:', len(sample))
    tasks = []
    with ThreadPoolExecutor(PARALLEL) as t:
        for i in range(0, len(sample), 3):
            tasks.append(t.submit(task, sample[i], 'equal'))
            tasks.append(t.submit(task, sample[i + 1], 'pure'))
            tasks.append(t.submit(task, sample[i + 2], 'pytha'))
    wait(tasks)
