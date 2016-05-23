# encoding: utf-8
"""Fit Random Forest model and tune its hyper-parameters."""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import easy_import, stats
from sklearn.ensemble import RandomForestClassifier
from itertools import product, cycle
import palettable
import random

if __name__ == "__main__":

    data_root = "../datasets"       # root path for datasets
    dataset = "quality_dataset.h5"  # set train dataset name
    n_fold = 5                     # number of folds for cross-validation
    do_default_params = True        # fit default random tree model
    do_grid_search = False           # hyper-parameter grid search
    figure_save = True              # save figure

    train_sizes = range(10000, 100000, 5000)

    print "Fit Random forest to", dataset
    path = os.path.join(data_root, dataset)

    # retrieve train dataset
    raw, filtered, labels = easy_import.extract_signals_train(path)

    # number of samples in the dataset
    n_samples = raw.shape[0]

    # zero-mean features
    raw_norm_f = raw - np.mean(raw, axis=0)
    filtered_norm_f = filtered - np.mean(filtered, axis=0)

    # zero-mean samples
    raw_norm_s = raw - np.mean(raw, axis=1)[..., np.newaxis]
    filtered_norm_s = filtered - np.mean(filtered, axis=1)[..., np.newaxis]

    # random forest model with (almost) default parameters
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=200)

    data = [[raw, "raw"],
            # [raw_norm_f, "raw normalized features"],
            [raw_norm_s, "raw normalized samples"],
            [filtered, "filtered"],
            # [filtered_norm_f, "filtered normalized features"],
            [filtered_norm_s, "filtered normalized samples"]]

    results_m = np.zeros([len(data), len(train_sizes)])
    results_s = np.zeros_like(results_m)


    for idx_size, size in enumerate(train_sizes):

        print "\n", "Training with", size, "samples"

        # select n samples among train samples
        selected = random.sample(range(n_samples), size)

        for item_idx, item in enumerate(data):

            x, name = item
            samples = x[selected, :]
            labels_sel = labels[selected]

            accuracy_m, accuracy_s = stats.report_cv_stats(n_fold, rf, samples, labels_sel, name)

            results_m[item_idx, idx_size] = accuracy_m
            results_s[item_idx, idx_size] = accuracy_s

    plt.figure()
    colors = palettable.tableau.TableauLight_10.mpl_colors

    for item_idx, item in enumerate(data):

        item_color = colors[item_idx]
        name = item[1]
        y_m = results_m[item_idx] * 100
        y_s = results_s[item_idx] * 100

        plt.fill_between(train_sizes, y_m - y_s, y_m + y_s,
                         facecolor=item_color, color=item_color, alpha=0.4)

        plt.plot(train_sizes, y_m, color=item_color, linewidth=2.5, alpha=0.8, label=name)

    # axis formatting
    # plt.ylim([75, 100])

    # title legend and labels
    plt.title("Model accuracy by train Size")
    plt.legend(loc="lower right", frameon=False)
    plt.ylabel(r"Accuracy $\mu\pm 2 \sigma$")
    plt.xlabel(r"Train Size")
    plt.tight_layout()

    if figure_save:
        plt.savefig("images/model_fit.png", format='png')

    plt.show()