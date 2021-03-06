# Dreem Challenge 21/05/2016

Data-driven EEG signal quality estimation.


## Introduction

The objective is to implement an algorithm that is able to assess the quality EEG signals, on the basis of a reference dataset of manually labelled samples.

The following steps will be performed:

* Dataset exploration: analysing the structure and properties of train and test datasets.

* Fitting a Random Forest model: training a plain random forest model with the train dataset and evaluating its performance.

* Hyper-parameter tuning: selecting random forest parameters in order to improve accuracy on the train set.

* EEG signal quality estimation: performing classification task on test datasets using the best performing random forest model.


## Dataset exploration

Three datasets are provided in [HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) format: 
* `quality_dataset.h5` is the train dataset with 137,030 samples. Each sample represents 2 `s` of a single EEG channel recording sampled at 250 `Hz` (yielding 500 data points per sample), and is labelled 0 for *bad quality* or 1 for *good quality*.
* `record1.h5` is a test dataset containing a single sample with 4 EEG channels and is partially labelled.
* `record2.h5` is a test dataset also containing a single sample with 4 EEG channels.

All datasets contain both *raw* samples and their *filtered* versions.

The structure and properties of the train and test datasets are analysed in greater detail in [Dataset exploration notebook](notebooks/dataset_exploration.ipynb).
The structure of the datasets (up to 2nd level) can be analysed with the script `python/dataset_exploration.py`.

**Example of raw and filtered samples from the train dataset**
![Grid Search](images/samples.png)

**Example of 'bad' and 'good' quality samples from the train dataset**
![Grid Search](images/samples_label.png)

**Histogram of 'raw' and 'filtered' samples from the train dataset**
![Grid Search](images/samples_hist.png)

**Histogram of 'raw' and 'filtered' samples from the train dataset with zero-mean normalization per sample**
![Grid Search](images/samples_hist_norm.png)

## Fitting a Random Forest model
Fitting a 'Random Forest' model with default parameters yields 94.75 (+/- 3.74) when training with raw signals,
and 88.71 (+/- 3.53) with filtered signals. The mean accuracy and 95% confidence interval are estimated with 20-fold cross-validation on the train dataset,
where 19/20 of the samples are used to fit the model and 1/20 to estimate the accuracy.

The performance of the model on the raw signals is further improved by scaling individual samples to have unit norm,
 which is a data pre-processing step called 'normalization'.

Although the filtered versions are visually appealing, the filtering process seems to discard information that is useful for the accuracy of the model.

**Sample output from `random_forest_fit_tune.py` (default model)**
```
Fit Random forest to quality_dataset.h5

Default model accuracy:
Accuracy (raw): 94.70 (+/- 3.85)
Accuracy (filtered): 88.66 (+/- 3.51)

Default model accuracy with normalized data:
```

## Hyper-parameter tuning
A random forest has several parameters that affect its accuracy. Popular parameters to tune are the number of trees,
the tree depth and maximum number of features for tree splits.
The script `random_forest_fit_tune.py` performs a grid search of these parameters on the normalized dataset.

**Sample output from `random_forest_fit_tune.py`**
```
Grid search for hyper-parameters:
Accuracy (raw, n_estimators=50, max_depth=25, max_features=5): 86.71 (+/- 2.27)
Accuracy (raw, n_estimators=50, max_depth=25, max_features=10): 86.55 (+/- 2.43)
Accuracy (raw, n_estimators=50, max_depth=25, max_features=15): 86.51 (+/- 2.45)
Accuracy (raw, n_estimators=50, max_depth=25, max_features=20): 86.59 (+/- 2.54)
Accuracy (raw, n_estimators=50, max_depth=50, max_features=5): 88.21 (+/- 0.48)
Accuracy (raw, n_estimators=50, max_depth=50, max_features=10): 88.17 (+/- 0.43)
Accuracy (raw, n_estimators=50, max_depth=50, max_features=15): 88.24 (+/- 0.39)
Accuracy (raw, n_estimators=50, max_depth=50, max_features=20): 88.14 (+/- 0.67)
Accuracy (raw, n_estimators=50, max_depth=100, max_features=5): 88.36 (+/- 0.29)
Accuracy (raw, n_estimators=50, max_depth=100, max_features=10): 88.26 (+/- 0.40)
Accuracy (raw, n_estimators=50, max_depth=100, max_features=15): 88.17 (+/- 0.62)
Accuracy (raw, n_estimators=50, max_depth=100, max_features=20): 88.15 (+/- 0.45)

etc...
```

![Grid Search](images/grid_search.png)

## EEG signal quality estimation

