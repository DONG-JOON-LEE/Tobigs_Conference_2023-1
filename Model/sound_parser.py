import librosa
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import IPython.display as ipd
import librosa.display
import scipy
import glob
import numpy as np
import math
import warnings
import pickle
from sklearn.utils import shuffle
import zipfile


class SoundParser: 
    def __init__(self) -> None:
        self.path_to_dataset = "./records/"
        self.train_tfrecords_filenames = glob.glob(os.path.join(self.path_to_dataset, 'train_*'))
        self.val_tfrecords_filenames = glob.glob(os.path.join(self.path_to_dataset, 'val_*'))


    def tf_record_parser(self, record):
        keys_to_features = {
            "noise_stft_phase": tf.io.FixedLenFeature((), tf.string, default_value=""),
            'noise_stft_mag_features': tf.io.FixedLenFeature([], tf.string),
            "clean_stft_magnitude": tf.io.FixedLenFeature((), tf.string)
        }

        features = tf.io.parse_single_example(record, keys_to_features)

        noise_stft_mag_features = tf.io.decode_raw(features['noise_stft_mag_features'], tf.float32)
        clean_stft_magnitude = tf.io.decode_raw(features['clean_stft_magnitude'], tf.float32)
        noise_stft_phase = tf.io.decode_raw(features['noise_stft_phase'], tf.float32)

        # reshape input and annotation images
        noise_stft_mag_features = tf.reshape(noise_stft_mag_features, (129, 8, 1), name="noise_stft_mag_features")
        clean_stft_magnitude = tf.reshape(clean_stft_magnitude, (129, 1, 1), name="clean_stft_magnitude")
        noise_stft_phase = tf.reshape(noise_stft_phase, (129,), name="noise_stft_phase")

        return noise_stft_mag_features, clean_stft_magnitude
    
    def get_data(self):
        train_dataset = tf.data.TFRecordDataset([self.train_tfrecords_filenames])
        train_dataset = train_dataset.map(self.tf_record_parser)
        train_dataset = train_dataset.shuffle(8192)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(512)
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.TFRecordDataset([self.val_tfrecords_filenames])
        test_dataset = test_dataset.map(self.tf_record_parser)
        test_dataset = test_dataset.repeat(1)
        test_dataset = test_dataset.batch(512)

        return train_dataset, test_dataset