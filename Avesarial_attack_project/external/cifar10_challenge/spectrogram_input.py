"""
Utilities for importing the Spectrogram dataset.

Each spectrogram in the dataset is a numpy array of shape (227, 169), with the values
being floats representing the spectrogram magnitudes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

class SpectrogramData(object):
    """
    Loads the Spectrogram dataset from .npy files.

    Inputs to constructor
    =====================

        - path: path to the folder containing spectrograms.npy and targets.npy files.
                The spectrograms should be of shape (N, 227, 169) and targets of shape (N,).
        - train_split: fraction of data to use for training (default 0.8).
                       The rest will be used for evaluation.
    """
    def __init__(self, path, train_split=0.8):
        spectrograms_file = os.path.join(path, 'spectrograms.npy')
        targets_file = os.path.join(path, 'targets.npy')
        
        # Load the data
        spectrograms = np.load(spectrograms_file)
        targets = np.load(targets_file)
        
        # Add channel dimension to match (227, 169, 1) format
        spectrograms = spectrograms[..., np.newaxis]
        
        # Shuffle the data
        n_samples = spectrograms.shape[0]
        indices = np.random.permutation(n_samples)
        spectrograms = spectrograms[indices]
        targets = targets[indices]
        
        # Split into train and eval
        n_train = int(n_samples * train_split)
        
        train_images = spectrograms[:n_train].astype('float32')
        train_labels = targets[:n_train].astype('int32')
        eval_images = spectrograms[n_train:].astype('float32')
        eval_labels = targets[n_train:].astype('int32')
        
        # Get number of classes
        num_classes = len(np.unique(targets))
        self.label_names = [f'Class_{i}' for i in range(num_classes)]
        
        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)
        
        print(f"Loaded {n_train} training samples and {n_samples - n_train} eval samples")
        print(f"Spectrogram shape: {spectrograms.shape[1:]}")
        print(f"Number of classes: {num_classes}")


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys