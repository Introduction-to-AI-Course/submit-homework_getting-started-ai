#-*- coding utf8 -*-
"""
@author:
@brief: create interaction feature
"""

from scipy import sparse
import numpy as np
from scipy.stats import pearsonr
from itertools import combinations, permutations
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import logging
import cofig
from utils import pickle

Columns = ['Pclass', 'Sex', 'Age', 'Embarked', 'Title', 'isAlone']

class Feature_Interaction:
    def __init__(self, dTrain, dTest, cols, logger, path, threshold=0.0001):
        self.cols = dTrain.columns
        self.unselected_cols = [col for col in self.cols if col not in cols]
        self.selected_cols = cols

        self.target = dTrain['Survived']

        self.selected_train = np.array(dTrain[self.selected_train])
        self.remained_train = np.array(dTrain[self.unselected_cols])
        self.selected_test = np.array(dTrain[self.unselected_cols])
        self.remained_test = np.array(dTest[self.unselected_cols])

        self.threshold = threshold
        self.path = path
        self.logger = logger

    def __str__(self):
        return "feature_interaction"

    def create_feature(self):
        turple_train_feature, turple_test_feature = self.create_tuples()
        triple_train_feature, triple_test_feature = self.create_triples()
        self.save(self.path+'/TUR/train.pkl', turple_train_feature)
        self.save(self.path+'/TUR/test.pkl', turple_test_feature)
        self.save(self.path+'/TRI/train.pkl', triple_train_feature)
        self.save(self.path+'/TRI/test.pkl', triple_test_feature)


    def create_tuples(self):
        self.logger.info("creating feature tuples")
        combination = list(combinations(self.selected_train.shape[1], 2))
        turple_train_feature = np.array([])
        turple_test_feature = np.array([])

        ori_score_cv = np.zeros(shape=(1, 5))
        kr = KFold(n_splits=5)
        train_one_hot = self.one_hot_encode(self.selected_train)

        for i, (train_idx, valid_idx)  in enumerate(kr.split(train_one_hot, self.target)):
            ori_score_cv[0, i] = self._get_score(train_one_hot[train_idx], train_one_hot[valid_idx],
                                                  self.target[train_idx], self.target[valid_idx])
        ori_score = np.mean(ori_score_cv, axis=1)
        del ori_score_cv, train_one_hot, kr

        for cmb in combination:
            turple_train_feature_tmp = self.selected_train[:, cmb[0]] + 10000 * self.selected_train[:, cmb[1]]
            turple_test_feature_tmp = self.selected_test[:, cmb[0]] + 10000 * self.selected_test[:, cmb[1]]
            new_train_ar = np.hstack([self.selected_train, turple_train_feature_tmp].reshape(-1, 1))
            new_train_one_hot = self.one_hot_encode(new_train_ar)
            score_cv = np.zeros(shape=(1, 5))
            kr = KFold(n_splits=5)

            for i, (train_idx, valid_idx) in enumerate(kr.split(new_train_one_hot, self.target)):
                score_cv[0, i] = self._get_score(new_train_one_hot[train_idx], new_train_one_hot[valid_idx], self.target[train_idx], self.target[valid_idx])
            score = np.mean(score_cv, axis=1)
            del score, score_cv, new_train_one_hot, new_train_ar
            if score -  ori_score > self.threshold:
                turple_feature_onr_hot = self.one_hot_encode(np.vstack([turple_train_feature_tmp, turple_test_feature_tmp]))
                turple_train_one_hot = turple_feature_onr_hot[:self.selected_train.shape[0], :]
                turple_test_one_hot = turple_feature_onr_hot[self.selected_train.shape[0]:, :]
                if turple_train_feature.shape[0] == 0:
                    turple_train_feature = turple_train_one_hot
                    turple_test_feature = turple_test_one_hot
                else:
                    turple_train_feature = np.hstack([turple_train_feature, turple_train_one_hot])
                    turple_test_feature = np.hstack([turple_test_feature, turple_test_one_hot])

        self.logger.info('feature turples done')
        return turple_train_feature, turple_test_feature

    def _get_score(self, xtrain, xvalid, ytrain, yvalid):
        model = SGDClassifier()
        model.fit(xtrain, ytrain)
        ypred = model.predict(xvalid)
        return roc_auc_score(yvalid, ypred)

    def create_triples(self):
        self.logger.info("creating feature triples")
        combination = list(combinations(self.selected_train.shape[1], 3))
        triple_train_feature = np.array([])
        triple_test_feature = np.array([])

        ori_score_cv = np.zeros(shape=(1, 5))
        kr = KFold(n_splits=5)
        train_one_hot = self.one_hot_encode(self.selected_train)

        for i, (train_idx, valid_idx) in enumerate(kr.split(train_one_hot, self.target)):
            ori_score_cv[0, i] = self._get_score(train_one_hot[train_idx], train_one_hot[valid_idx],
                                                 self.target[train_idx], self.target[valid_idx])
        ori_score = np.mean(ori_score_cv, axis=1)
        del ori_score_cv, train_one_hot, kr


        for cmb in combination:
            triple_train_feature_tmp = self.selected_train[:, cmb[0]] + self.selected_train[:, cmb[1]]  * 100 + self.selected_train[:, cmb[2]] * 10000
            triple_test_feature_tmp = self.selected_test[:, cmb[0]] + self.selected_test[:, cmb[1]] * 100 + self.selected_test[:, cmb[2]] * 10000
            new_train_ar = np.hstack([self.selected_train, triple_train_feature_tmp].reshape(-1, 1))
            new_train_one_hot = self.one_hot_encode(new_train_ar)
            score_cv = np.zeros(shape=(1, 5))
            kr = KFold(n_splits=5)

            for i, (train_idx, valid_idx) in enumerate(kr.split(new_train_one_hot, self.target)):
                score_cv[0, i] = self._get_score(new_train_one_hot[train_idx], new_train_one_hot[valid_idx],
                                                 self.target[train_idx], self.target[valid_idx])
            score = np.mean(score_cv, axis=1)
            del score, score_cv, new_train_one_hot, new_train_ar
            if score - ori_score > self.threshold:
                triple_feature_onr_hot = self.one_hot_encode(
                    np.vstack([triple_train_feature_tmp, triple_test_feature_tmp]))
                triple_train_one_hot = triple_feature_onr_hot[:self.selected_train.shape[0], :]
                triple_test_one_hot = triple_feature_onr_hot[self.selected_train.shape[0]:, :]
                if triple_train_feature.shape[0] == 0:
                    triple_train_feature = triple_train_one_hot
                    triple_test_feature = triple_test_one_hot
                else:
                    triple_train_feature = np.hstack([triple_train_feature, triple_train_one_hot])
                    triple_test_feature = np.hstack([triple_test_feature, triple_test_one_hot])

        self.logger.info('feature triple done')

        return triple_train_feature, triple_test_feature

    def one_hot_encode(self, X):
        return OneHotEncoder().fit_transform(X)

    def save(self, fname, data):
        pickle._save(fname, data)

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
    Feature_Interaction(dTrain=dTrain, dTest=dTest, cols=Columns, logger=logger, path=cofig.INTERA_DATA_DIR).create_feature()

if __name__ == '__main__':
    main()


