# encoding: utf-8
"""Basic structure of train and test datasets (up to 2 level).
"""

import h5py
import os


def print_hdf5_data_shape(path):
    """Print data shape of HDF5 dataset (up to 2nd level).
    Input
    =====
    path: path of HDF5 file

    Output
    ======
    Prints data shape of 1st level datasets and level datasets within 1st level groups.
    """

    # check if path exists or raise error otherwise
    if os.path.exists(path):

        with h5py.File(path, 'r') as f:

            print 'Dataset', path, 'contains:'

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

    else:
        raise ValueError('Cannot open HDF5 file:', path)


if __name__ == "__main__":

    # root path for datasets and list of datasets
    datasets_root = '../../datasets'
    datasets = ["quality_dataset.h5", "record1.h5", "record2.h5"]

    for item in datasets:
        path = os.path.join(datasets_root, item)

        print_hdf5_data_shape(path)