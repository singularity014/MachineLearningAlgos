# import libs
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt


def data_loader(data_path, split=True):

    """
    Takes data path, reads through pandas, returns df and/or features, labels
    :param data_path: path to data.
    :param split: if True, returns matrix of label and features along with dataframe.
                  Only DataFrame otherwise
    :return: dataframe, feature matrix, labels
    """
    # load dataset -
    dataset = pd.read_csv(data_path)
    X = None
    y = None
    if split:
        # X (feature matrix)
        X = dataset.iloc[:, :-1].values
        # y (Label array)
        y = dataset.iloc[:, -1].values
    return dataset, X, y


def missing_data_handler(X):
    """
    Takes feature matrix and handles msising data..
    :param X: feature matrix to be imputed..
    :return:
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # takes only numerical features, exclude string columns
    print(X)
    imputer.fit(X[:, [1, 2]])
    X[:, [1, 2]] = imputer.transform(X[:, [1, 2]])
    print('After imputing (Mean)...')
    print(X)
    return X


if __name__ == "__main__":

    dat_path = "../DataSets/pre_processing_sample.csv"
    df, X, y = data_loader(dat_path)

    missing_data_handler(X)
