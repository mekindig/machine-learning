# wine
# Mark Kindig
# Copied liberally from:
# Classification and Clustering of Wine Dataset
#  and: https://github.com/georgetown-analytics/machine-learning/blob/master/notebook/Wheat%20Classification.ipynb
# Author:   Author:   Benjamin Bengfort <bbengfort@districtdatalabs.com>
# Created:  Thu Feb 26 17:56:52 2015 -0500
#
# Copyright (C) 2015 District Data Labs
# For license information, see LICENSE.txt
#
# ID: wheat.py [] benjamin@bengfort.com $

"""
Classification and Clustering of Wine Dataset
"""

##########################################################################
## Imports
##########################################################################

import time

from utils import *

def load_wine():
    return load_data('wine')
##########################################################################
## Wine Kernel Classification
##########################################################################

if __name__ == '__main__':

    start_time = time.time()

    # Load the dataset
    # Save dataset as a variable we can use.
    dataset    = load_wine()
    data       = dataset.data
    print dataset.data.shape
    target     = dataset.target
    print dataset.target.shape

    # Perform SVC Classification
    fit_and_evaluate(dataset, SVC, "Red Wine SVM Classifier")

    # Perform kNN Classification
    fit_and_evaluate(dataset, KNeighborsClassifier, "Read Wine kNN Classifier", n_neighbors=12)
