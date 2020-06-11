#-*- coding utf8 -*-
"""
@author:
@brief:create the statistical feature
"""

import numpy as np
from scipy.stats import pearsonr

import logging
from _collections import defaultdict

import cofig
from utils import pickle

methods = [
    'count',
    'ratio',
    "poly"
]

cols = ['Pclass', 'Sex', 'Age', 'IsAlone', 'Age*Class']

class Stats_feature:
    def __init__(self, dTrain, dTest, cols, logger, path, methods=methods, threshold=0.01):
        self.cols = dTrain.columns
        self.selected_col = cols
        self.unselected_col = [col for col in self.cols if col not in cols]

        self.selected_train = np.array(dTrain[self.selected_col])
        self.selected_test = np.array(dTest[self.selected_col])
        self.unselected_train = np.array(dTrain[self.unselected_col])
        self.unselected_test = np.array(dTest[self.unselected_col])
        self.target = dTrain['Survived']

        self.methods = methods
        self.threshold = threshold

        self.path = path
        self.logger = logger

    def create_feature(self):
        self.logger.info("create polynomial feature")

        assert self.methods

        col_value_count = None
        for method in self.methods:
            stats_train_feature = np.array([])
            stats_test_feature = np.array([])

            if method == 'count':
                self.logger.info("start creating count feature")
                col_value_count = self._cal_count()
                stats_train_feature_temp  = np.zeros(shape=(self.selected_train.shape[0], 1))
                stats_test_feature_temp = np.zeros(shape=(self.selected_test.shape[0], 1))
                for col in range(self.selected_col):
                    for row in range(self.selected_train.shape[0]):
                        stats_train_feature_temp[row, 0] = col_value_count[self.selected_train[row, col]]
                    if_create = self._comp_pearsonr(stats_train_feature_temp[:, col], self.target)
                    if if_create:
                        if stats_train_feature.shape[0] == 0:
                            stats_train_feature = stats_train_feature_temp
                        else:
                            stats_train_feature = np.hstack([stats_train_feature, stats_test_feature_temp])
                        for row in range(self.selected_test.shape[0]):
                            stats_test_feature_temp[row, col] = col_value_count[self.selected_test[row, col]]
                        if stats_test_feature.shape[0] == 0:
                            stats_test_feature = stats_test_feature_temp
                        else:
                            stats_test_feature = np.hstack([stats_test_feature, stats_test_feature_temp])
                self.save(self.path+'/count/train.pkl', stats_train_feature)
                self.save(self.path+'/count/test.pkl', stats_test_feature)
                self.logger.info("count feature done")
                continue

            if method == 'ratio':
                self.logger.info("start creating ratio feature")
                if not col_value_count:
                    col_value_count = self._cal_count()
                col_value_ratio = self._cal_ratio(col_value_count)
                stats_test_feature_temp = np.zeros(shape=(self.selected_test.shape[0], 1))
                stats_train_feature_temp = np.zeros(shape=(self.selected_train.shape[0], 1))
                for col in range(self.selected_train.shape[1]):
                    for row in range(self.selected_train.shape[0]):
                        stats_train_feature_temp[row, 0] = col_value_ratio[self.selected_train[row, col]]
                    if_create = self._comp_pearsonr(stats_train_feature_temp[:, 0], self.target)
                    if if_create:
                        if stats_train_feature.shape[0] == 0:
                            stats_train_feature = stats_train_feature_temp
                        else:
                            stats_train_feature = np.hstack([stats_train_feature, stats_test_feature_temp])
                        for row in range(self.selected_test.shape[0]):
                            stats_test_feature_temp[row, col] = col_value_count[self.selected_test[row, col]]
                        if stats_test_feature.shape[0] == 0:
                            stats_test_feature = stats_test_feature_temp
                        else:
                            stats_test_feature = np.hstack([stats_test_feature, stats_test_feature_temp])
                self.save(self.path+'/ratio/train.pkl', stats_train_feature)
                self.save(self.path+'/ratio/test.pkl', stats_test_feature)
                self.logger.info("ratio feature done")
                continue

            if method == 'poly':
                self.logger.info("start poly feature")
                poly_train, poly_test = self._cal_poly()
                poly_feature_train = np.array([])
                poly_feature_test = np.array([])
                for col in range(self.selected_col):
                    if_create = self._comp_pearsonr(poly_train[:, col], self.target)
                    if if_create:
                        if poly_feature_train.shape[0] == 0:
                            poly_feature_train = poly_train[:, col].reshape(-1, 1)
                            poly_feature_test = poly_test[:, col].reshape(-1, 1)
                        else:
                            poly_feature_train = np.hstack([poly_feature_train, poly_train[:, col].reshape(-1, 1)])
                            poly_feature_test = np.hstack([poly_feature_test, poly_test[:, col].reshape(-1, 1)])
                self.save(self.path+'/poly/train.pkl', poly_feature_train)
                self.save(self.path+'/poly/test.pkl', poly_feature_test)
                self.logger.info('poly done')



    def _comp_pearsonr(self, X, y):
        return pearsonr(X, y)[0] > self.threshold

    def _cal_count(self):
        value_count = defaultdict(int)
        col_value_count = [value_count * len(self.selected_col)]

        data = np.vstack([self.selected_train, self.selected_test])

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                col_value_count[i][data[i, j]] += 1

        return col_value_count

    def _cal_ratio(self, col_value_count):
        value_ratio = defaultdict(float)
        col_value_ratio = [value_ratio * len(self.selected_col)]

        data = np.vstack([self.selected_train, self.selected_train])

        for col in range(data.shape[0]):
            col_sum = np.sum(data, axis=0)
            for (value, counts) in col_value_count[col]:
                col_value_ratio[col][value] = float(counts) / float(col_sum)

        return col_value_ratio

    def _cal_poly(self):
        return np.power(self.selected_train, 2), np.power(self.selected_test, 2)

    def save(self, filename, data):
        pickle._save(filename, data)


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="mantian.log", filemode='a', level=logging.DEBUG,
                        datefmt="%m/%d/%y %H:%M:%S")
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt="%m/%d/%y %H:%M%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logger = logging.getLogger(__name__)

    dTrain = pickle._load(cofig.CLEAN_TRAIN_DATA)
    dTest = pickle._load(cofig.CLEAN_TEST_DATA)

    Stats_feature(dTrain=dTrain, dTest=dTest, cols=cols, logger=logger, path=cofig.STATS_DATA_DIR).create_feature()



if __name__ == '__main__':
    main()


