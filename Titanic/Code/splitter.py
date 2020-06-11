#-*- coding utf8 -*-
"""
@author:
@brief: splitter for data
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.model_selection import ShuffleSplit

import cofig
from utils import pickle

class StratifiedShuffleSplit(StratifiedShuffleSplit):
    def __init__(self, y, n_iter=10, test_size=0.1, train_size=None,
                 random_state=None):
        n = len(y)
        self.y = np.array(y)
        self.classes, self.y_indices = np.unique(y, return_inverse=True)
        self.random_state = random_state
        self.train_size = train_size
        self.test_size = test_size
        self.n_iter = n_iter
        self.n = n
        self.n_train, self.n_test = _validate_shuffle_split(n, test_size, train_size)


class Splitter:
    def __init__(self, dfTrain, dfTest, n_iter=5, random_state=cofig.RANDOM_SEED,
                 verbose=False):
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

    def __str__(self):
        return "Splitter"


    def _get_df_idx(self, df, col, values):
        return np.where(df[col].isin(values))[0]

    def split(self):

        if self.verbose:
            print("*" * 50)
            print("Original Train and Test Split")

        if self.verbose:
            print("*" * 50)
            print("Split")
        rs = ShuffleSplit(n=self.dfTrain.shape[0], n_iter=1, test_size=0.69, random_state=self.random_state)
        for trainInd, validInd in rs:
            dfTrain2 = self.dfTrain.iloc[trainInd].copy()
            dfValid = self.dfTrain.iloc[validInd].copy()

        return self

    def save(self, fname):
        pickle._save(fname, self.splits)


def main():
    dtrain = pd.read_csv(cofig.TRAIN_DATA, encoding='utf8')
    dtest = pd.read_csv(cofig.TEST_DATA, encoding='utf8')

    #split
    splitter = Splitter(dfTrain=dtrain,
                                dfTest=dtest,
                                n_iter=config.N_RUNS,
                                random_state=config.RANDOM_SEED,
                                verbose=True,
                        )
    splitter.split()
    splitter.save("%s/splits_level1.pkl"%config.SPLIT_DIR)

if __name__ == '__main__':
    main()
