# encoding: utf-8
"""Basic structure of train and test datasets (up to 2 level)."""

import h5py
import os
from utils import Config

# list of datasets
records = ["quality_dataset.h5", "record1.h5", "record2.h5"]

# dataset root path
config = Config()
config.data_root = '../datasets'

for record in records:
    path = os.path.join(config.data_root, record)

    # retrieve dataset
    f = h5py.File(path, 'r')

    print 'Dataset', record, 'contains:'

    # iterate members
    for item in f.items():
        name, obj = item

        # print data shape if member is an h5py Dataset
        if isinstance(obj, h5py.Dataset):

            print '  ', name + ':', obj.shape

        # iterate children and print data shape if member is an h5py Group
        elif isinstance(obj, h5py.Group):

            for member in obj:
                print '  ', name + '/' + member + ':', f[name][member].shape

        # raise error if neither dataset nor group
        else:
            raise ValueError('Cannot identify HDF5 item:', type(obj))

    print