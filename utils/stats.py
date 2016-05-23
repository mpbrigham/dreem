# encoding: utf-8
"""Helper functions for retrieving model training statistics."""

from sklearn import cross_validation


def report_cv_stats(n_fold, model, samples, labels, comment=None):
    """Compute mean and standard deviation of model accuracy with n-fold cross validation.

    Input
    =====
    n_fold: number of folds (train on (n_fold-1)/n_fold samples)
    model: sklearn model
    samples: samples in rows
    labels: labels for each sample row
    comment: comment to display

    Output
    ======
    accuracy_m: (normalized) accuracy mean
    accuracy_s:  (normalized) accuracy standard deviation
    Prints accuracy mean and standard deviation of model in percentage.
    """

    # compute n-fold cross validation accuracy for model
    accuracy = cross_validation.cross_val_score(model, samples, labels, cv=n_fold)

    # compute mean and standard deviation
    accuracy_m = accuracy.mean()
    accuracy_s = accuracy.std()

    text = ""
    if comment:
        text = "(" + comment + ")"

    print("Accuracy" + text + ": %0.2f (+/- %0.2f)" % (accuracy_m * 100, accuracy_s * 100 * 2))

    return accuracy_m, accuracy_s