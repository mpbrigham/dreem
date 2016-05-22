# encoding: utf-8
"""Fit Random Forest model and tune its hyper-parameters."""

import os
from utils import easy_import, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from itertools import product


if __name__ == "__main__":

    # root path for datasets
    data_root = "../datasets"

    # number of folds for cross-validation
    n_fold = 20

    # number of samples used in train or None to use all samples
    limit_train_n = None

    # define train dataset
    dataset = "quality_dataset.h5"

    print "Fit Random forest to", dataset
    path = os.path.join(data_root, dataset)

    # retrieve train dataset
    raw, filtered, labels = easy_import.extract_signals_train(path)

    # train with reduced datasets
    if limit_train_n:
        raw = raw[:limit_train_n, :]
        filtered = filtered[:limit_train_n, :]
        labels = labels[:limit_train_n]

    # normalized versions of signals
    raw_n = normalize(raw)
    filtered_n = normalize(filtered)

    # random forest model with default parameters
    rf = RandomForestClassifier(n_jobs=-1)

    # print "\n", "Default model accuracy:"
    # stats.report_cv_stats(n_fold, rf, raw, labels, "raw")
    # stats.report_cv_stats(n_fold, rf, filtered, labels, "filtered")

    print "\n", "Default model accuracy with normalized data:"
    stats.report_cv_stats(n_fold, rf, raw_n, labels, "raw norm")
    stats.report_cv_stats(n_fold, rf, filtered_n, labels, "filtered norm")

    # grid search for hyper-parameters
    print "\n", "Grid search for hyper-parameters:"
    for n_estimators, max_depth, max_features in product([50, 100, 150, 200],
                                                         [25, 50, 100, 200],
                                                         [5, 10, 15, 20]):

        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    n_jobs=-1)

        desc = "raw, n_estimators=" + str(n_estimators) \
               + ", max_depth=" + str(max_depth) \
               + ", max_features=" + str(max_features)

        stats.report_cv_stats(n_fold, rf, raw_n, labels, desc)
