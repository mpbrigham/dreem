# encoding: utf-8
"""Helper functions for retrieving model training statistics."""

from sklearn import cross_validation


def report_cv_stats(n_fold, model, x, y, name):
    """Report n-fold cross validation accuracy for sklearn model.

    Input
    =====
    n_fold: path to a record
    model: sklearn model
    x: train data
    y: labels
    name: display text

    Output
    ======
    prints n-fold cross validation accuracy for model.
    """
    scores = cross_validation.cross_val_score(model, x, y, cv=n_fold)
    print("Accuracy (" + name + "): %0.2f (+/- %0.2f)" % (scores.mean() * 100, scores.std() * 100 * 2))