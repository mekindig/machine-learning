# Mark Kindig
"""
This file creates the meta.json and pandas dataset.txt files from the
winequality-red.csv file that was downloaded from the UCI machine learning repo.
"""

# Imports for Data Wrangling and Extraction
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import radviz

DIRNAME   = os.path.dirname(__file__)
DATAPATH  = os.path.join(DIRNAME, "winequality-red.csv")
OUTPATH   = os.path.join(DIRNAME, "dataset.txt")

# last feature, "label", is quality assessed by oenologist
FEATURES  = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfer dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "label"
]

LABEL_MAP = {
    1: "3",
    2: "4",
    3: "5",
    4: "6",
    5: "7",
    6: "8",
    7: "9"
}

if __name__ == "__main__":
    
    # Read the data into a DataFrame
    df = pd.read_csv(DATAPATH, delimiter=';', header=0, names=FEATURES)
    
    # Convert class labels into text.  I don't think this was actually necessary because data had labels (which I skipped)
    #for k,v in LABEL_MAP.items():
    #    df.ix[df.label == k, 'label'] = v
 
    # Save CSV data file -- definitely cleaner file now
    df.to_csv(OUTPATH, sep='\t', index=False, header=False)

    print "Wrote dataset of %i instances and %i attributes to %s" % (df.shape + (OUTPATH,))

    with open('meta.json', 'w') as f:
        meta = {'feature_names': FEATURES, 'target_names': LABEL_MAP}
        json.dump(meta, f, indent=4)

    # Describe the dataset
    print df.describe()

    # Determine the shape of the data
    print "{} instances with {} features\n".format(*df.shape)

    # Determine the frequency of each class
    print df.groupby('label')['label'].count()

    # Create a scatter matrix of the dataframe features
    scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
    plt.show()

    # Parallel Coordinates
    plt.figure(figsize=(12,12))
    parallel_coordinates(df,'label')
    plt.show()

    # Radviz
    plt.figure(figsize=(12,12))
    radviz(df, 'label')
    plt.show()
