# encoding: utf-8
"""Helper functions for reading datastes in HDF5 format."""

import h5py
import numpy as np


def extract_signals(record):
    """Open a record file and extract raw and filtered signals.

    Input
    =====
    record: path to a record

    Output
    ======
    raw: raw signal over 4 channels
    filtered: filtered signal over 4 channels
    """

    # Open an h5 file
    f = h5py.File(record, 'r')

    # Extract raw signal
    raw = np.array([f['channel1/raw'][:],
                     f['channel2/raw'][:],
                     f['channel3/raw'][:],
                     f['channel4/raw'][:]])

    # Extract Filtered signal
    filtered = np.array([f['channel1/visualization'][:],
                         f['channel2/visualization'][:],
                         f['channel3/visualization'][:],
                         f['channel4/visualization'][:]])

    return raw, filtered

def extract_signals_train(record):
    """Open a train record file and extract raw and filtered signals, with corresponding labels.

    Input
    =====
    record: path to a train record

    Output
    ======
    raw: raw signals over 1 channel
    filtered: filtered signals over 1 channel
    labels: labels
    """

    # Open an h5 file
    f = h5py.File(record, 'r')

    # Extract raw signal
    raw = np.array(f['dataset'][:, :500])

    # Extract filtered signal
    filtered = np.array(f['dataset'][:, 500:1000])

    # Extract labels
    labels = np.array(f['dataset'][:, 1000])

    return raw, filtered, labels