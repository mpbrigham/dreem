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
    limit_train_n = 20000           # number of samples or None to use them all
    figure_save = True              # save figure

    # hyper-parameters range
    n_estimators = [100, 200, 300]
    max_depth = [50, 100, 150]
    max_features = [10, 20, 30]

    # select dataset idx among:
    # 0=raw, 1=raw normalized features, 2=raw normalized samples
    # 3=filtered, 4=filtered normalized features, 5=filtered normalized samples
    dataset_idx = 2

    dataset_name = {0: "raw", 1: "raw normalized features", 2: "raw normalized samples",
                    3: "filtered", 4: "filtered normalized features", 5: "filtered normalized samples"}

    print "Grid search hyper-parameters for", dataset

    print "\n", "Type:", dataset_name[dataset_idx]

    # retrieve train dataset
    path = os.path.join(data_root, dataset)
    raw, filtered, labels = easy_import.extract_signals_train(path)

    # number of samples in the dataset
    n_samples = raw.shape[0]

    # reduced train dataset if limit_train_n is set
    if limit_train_n and limit_train_n <= n_samples:
        print "Reduced dataset:", limit_train_n, "samples"

        indexes = random.sample(range(n_samples), limit_train_n)
        raw = raw[indexes, :]
        filtered = filtered[indexes, :]
        labels = labels[indexes]

    if dataset_idx == 0:
        samples = raw

    elif dataset_idx == 1:
        samples = raw - np.mean(raw, axis=0)

    elif dataset_idx == 2:
        samples = raw - np.mean(raw, axis=1)[..., np.newaxis]

    elif dataset_idx == 3:
        samples = filtered

    elif dataset_idx == 4:
        samples = filtered - np.mean(filtered, axis=0)

    elif dataset_idx == 5:
        samples = filtered - np.mean(filtered, axis=1)[..., np.newaxis]

    else:
        raise ValueError('Unknown dataset type:', dataset_idx)

    # plot markers and figure spacing variables
    markers = ("o", "v", "^", "<", ">", "8", "s", "p", "*", "h", "H", "D", "d")
    evals = len(max_depth) * len(max_features)
    x_step = np.min(np.diff(n_estimators)) / evals / 2
    x_step_start = x_step * evals / 2

    print
    plt.figure()

    for idx, x in enumerate(n_estimators):

        # reset marker and coilor cycles
        markercycler = cycle(markers)
        plt.gca().set_color_cycle(palettable.tableau.Tableau_10.mpl_colors)

        # reset figure positioning and labels
        pos = -x_step/2
        label_str = None

        for y, z in product(max_depth, max_features):

            # fit Random Forest model with n_estimators=x, max_depth=y, max_features=z
            rf = RandomForestClassifier(n_jobs=-1,
                                        n_estimators=x, max_depth=y, max_features=z)

            # compute accuracy mean and std dev with cross validation
            desc = "n_estimators=" + str(x) + ", max_depth=" + str(y) + ", max_features=" + str(z)
            scores_m, scores_s = stats.report_cv_stats(n_fold, rf, samples, labels, desc)

            pos += x_step

            # add label if first estimator
            if idx == 0:
                label_str = "d=" + str(y) + " f=" + str(z)

            plt.errorbar(x - x_step_start + pos, scores_m*100, scores_s*100,
                         marker=next(markercycler), label=label_str)

    # axis formatting
    plt.xlim([min(n_estimators) - 2*x_step_start,
              max(n_estimators) + 2*x_step_start])
    plt.ylim([80, 100])
    plt.xticks(n_estimators)
    plt.yticks(range(50, 100, 10))

    # title legend and labels
    plt.title("Hyper-parameters Grid Search")
    plt.legend(loc="lower right", numpoints=1)
    plt.ylabel(r"Accuracy $\mu\pm 2 \sigma$")
    plt.xlabel(r"n_estimators")
    plt.tight_layout()

    if figure_save:
        plt.savefig("images/grid_search.png", format='png')

    plt.show()