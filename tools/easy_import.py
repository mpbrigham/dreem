# encoding: utf-8
import h5py
import numpy as np


def extract_signals(record):
    """Open a record file and extract raw and filtered signals."""
    # Open an h5 file
    f = h5py.File(record, 'r')

    # Extract raw signal
    raws = np.array([f['channel1/raw'][:],
                     f['channel2/raw'][:],
                     f['channel3/raw'][:],
                     f['channel4/raw'][:]])

    # Extract Filtered signal
    filtered = np.array([f['channel1/visualization'][:],
                         f['channel2/visualization'][:],
                         f['channel3/visualization'][:],
                         f['channel4/visualization'][:]])

    return raws, filtered
