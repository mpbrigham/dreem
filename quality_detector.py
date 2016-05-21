# encoding: utf-8
"""On this file you will implement you quality predictor function."""

import numpy as np
import os
from utils import easy_import
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import multiprocessing

class Config:
    """Configuration settings."""
    def __init__(self):
        self.data_root = None

def predict_quality(record):
    """Predict the quality of the signal.

    Input
    =====
    record: path to a record

    Output
    ======
    results: a list of 4 signals between 0 and 1 estimating the quality
    of the 4 channels of the record at each timestep.
    This results must have the same size as the channels.
    """
    # Extract signals from record
    raws, filtered = easy_import.extract_signals(record)
    # Initialize results (same size as the channels)
    results = np.zeros((raws.shape))

    return results


if __name__ == "__main__":

    config = Config()
    config.data_root = '../datasets'
    nprocs = multiprocessing.cpu_count()

    record = os.path.join(config.data_root, "quality_dataset.h5")

    raws, filtered, labels = easy_import.extract_data(record)

    for item in [10, 50, 100, 200, 300, 400, 500]:

        print 'n_estimators=', item

        rf = RandomForestClassifier(n_estimators=item, n_jobs=nprocs)

        scores = cross_validation.cross_val_score(rf, raws, labels, cv=5)

        print("Accuracy (raw): %0.2f (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 100 * 2))

        # scores = cross_validation.cross_val_score(rf, filtered, labels, cv=5)
        #
        # print("Accuracy (filt): %0.2f (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 100 * 2))

        # record = os.path.join(config.data_root, "record1.h5")
        # results = predict_quality(record)
