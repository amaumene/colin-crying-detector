#!/usr/bin/env python3

import os
import getopt
# audio stuff
import pyaudio
import wave
import time
# fastai
from fastai.basic_train import *
from fastai.vision import *
# spectrogram
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
# pushover
from pushover import init, Client

class Recorder(object):
    def __init__(self, channels=1, rate=44100, frames_per_buffer=8192):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                self.frames_per_buffer)

class RecordingFile(object):
    def __init__(self, fname, mode, channels,
            rate, frames_per_buffer):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.frames_per_buffer,
                stream_callback=self.get_callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def get_callback(self):
        def callback(in_data, frame_count, time_info, status):
            self.wavefile.writeframes(in_data)
            return in_data, pyaudio.paContinue
        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

def generate_spectrogram(filepath):
    data, sampling_rate = librosa.load(filepath)
    plt.figure(figsize=(1, 1))
    plt.axis('off')
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    librosa.display.specshow(librosa.power_to_db(melspectrogram, ref=np.max))
    img_filepath = Path(str(filepath).replace('.wav', '.png'))
    plt.savefig(img_filepath, dpi=224)
    plt.close()
    return img_filepath

def main(argv):
    apikey = ''
    userkey = ''
    try:
        opts, args = getopt.getopt(argv,"ha:u:",["api-key=","user-key="])
    except getopt.GetoptError:
        print('detect.py -a <api-key> -u <user-key>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('detect.py -a <api-key> -u <user-key>')
            sys.exit()
        elif opt in ("-a", "--api-key"):
            apikey = arg
        elif opt in ("-u", "--user-key"):
            userkey = arg
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
    data_bunch = ImageDataBunch.single_from_classes(path, [0, 1],
            ds_tfms=get_transforms(do_flip=False, max_rotate=0., max_lighting=0., max_warp=0.),
            size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load('stage-2')
    rec = Recorder(channels=1)

    #init pushover
    init(apikey)
    while True:
        with rec.open('/tmp/file.wav', 'wb') as recfile2:
            recfile2.start_recording()
            time.sleep(5.0)
            recfile2.stop_recording()
        img = open_image(generate_spectrogram('/tmp/file.wav'))
        is_crying = int(learn.predict(img)[0])
        print("Colin status:", is_crying)
        if is_crying == 1:
            Client(userkey).send_message("Colin is crying :(", title="Colin")

if __name__ == '__main__':
    main(sys.argv[1:])
