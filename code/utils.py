# utils
# Mark Kindig
# Utility functions for handling data
#
# Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Feb 26 17:47:35 2015 -0500
#
# Copyright (C) 2015 District Data Labs
# For license information, see LICENSE.txt
#
# ID: utils.py [] benjamin@bengfort.com $

"""
Utility functions for handling data
Updated from Machine Learning/Wheat Classification iPython Notebook
"""

##########################################################################
## Imports
##########################################################################

# Imports for prepping dataset
import os
import csv
import time
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.datasets.base import Bunch

# Imports for Classification
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import KFold

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM


##########################################################################
## Module Constants
##########################################################################

BASE_DIR = os.path.abspath(os.path.join(".",".."))
DATA_DIR = os.path.abspath(os.path.join(".","..","data"))
CODE_DIR = os.path.join(BASE_DIR, "code")

##########################################################################
## Dataset Loading
##########################################################################


def load_data(path, descr=None, target_index=-1):
    """
    Returns a scklearn dataset Bunch which includes several important
    attributes that are used in modeling:

        data: array of shape n_samples * n_features
        target: array of length n_samples
        feature_names: names of the features
        target_names: names of the targets
        filenames: names of the files that were loaded
        DESCR: contents of the readme

    This data therefore has the look and feel of the toy datasets.

    Pass in a path usually just the name of the location in the data dir.
    It will be joined with the result of `get_data_home`. The contents are:

        path
            - README.md     # The file to load into DESCR
            - meta.json     # A file containing metadata to load
            - winequality-red/white.csv   # The downloaded input file (split?)
            - dataset.csv   # The pandas read_csv file

    You can specify another descr, another feature_names, and whether or
    not the dataset has a header row. You can also specify the index of the
    target, which by default is the last item in the row (-1)
    """

    root          = os.path.join(DATA_DIR, path)

    # Show the contents of the data directory
    for name in os.listdir(root):
        if name.startswith("."): continue
        print "- {}".format(name)

    # Construct the 'BUnch' for the Wine dataset
    filenames     = {
        'meta': os.path.join(root, 'meta.json'),
        'rdme': os.path.join(root, 'README.md'),
        'data': os.path.join(root, 'dataset.txt'),
    }

    target_names  = None
    feature_names = None
    DESCR         = None

    # Load the meta data from the meta json
    with open(filenames['meta'], 'r') as f:
        meta = json.load(f)
        target_names  = meta['target_names']
        feature_names = meta['feature_names']

    # Load the description from the README
    with open(filenames['rdme'], 'r') as f:
        DESCR = f.read()

    # Load the dataset from the text/csv file.
    dataset = np.loadtxt(filenames['data'])
    
    # Extract the target from the data
    data    = dataset[:, 0:-1]
    target  = dataset[:, -1]

    # Target assumed to be either last or first column
    if target_index == -1:  
        data   = dataset[:, 0:-1]  # In last column
        target = dataset[:, -1]
    elif target_index == 0:    # In first column
        data   = dataset[:, 1:]
        target = dataset[:, 0]
    else:
        raise ValueError("Target index must be either -1 or 0")

    return Bunch(data=data,
                 target=target,
                 filenames=filenames,
                 target_names=target_names,
                 feature_names=feature_names,
                 DESCR=DESCR)

def load_wine():
    return load_data('wine')

# Classification function
def fit_and_evaluate(dataset, model, label, **kwargs):
    # Try NNs, SVMs
    start = time.time()
    scores = {'precision':[], 'recall':[], 'accuracy':[], 'f1':[]}


    for train, test in KFold(dataset.data.shape[0], n_folds=12, shuffle=True):
        X_train, X_test = dataset.data[train], dataset.data[test]
        y_train, y_test = dataset.target[train], dataset.target[test]
        
        estimator = model(**kwargs)
        estimator.fit(X_train, y_train)
        
        expected  = y_test
        predicted = estimator.predict(X_test)
        
        # Append our scores to the tracker
        scores['precision'].append(metrics.precision_score(expected, predicted, average="weighted"))
        scores['recall'].append(metrics.recall_score(expected, predicted, average="weighted"))
        scores['accuracy'].append(metrics.accuracy_score(expected, predicted))
        scores['f1'].append(metrics.f1_score(expected, predicted, average="weighted"))

    # Report
    print "Build and Validation of {} took {:0.3f} seconds".format(label, time.time()-start)
    print "Validation scores are as follows:\n"
    print pd.DataFrame(scores).mean()
    
    # Write official estimator to disk
    estimator = model(**kwargs)
    estimator.fit(dataset.data, dataset.target)
    
    outpath = label.lower().replace(" ", "-") + ".pickle"
    with open(outpath, 'w') as f:
        pickle.dump(estimator, f)

    print "\nFitted model written to:\n{}".format(os.path.abspath(outpath))

