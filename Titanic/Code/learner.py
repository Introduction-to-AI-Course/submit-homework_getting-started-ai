#-*- coding utf8 -*-
"""
@author:
@brief:
      learner model for ensemble selection
"""

import time
import os
import sys
from  optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from sklearn.metrics import roc_auc_score

from utils.xgb_utils import XGBClassifier
from utils.skl_utils import SVC, LinearSVC, AdaboostClassifier, RandomLogistic
from utils import pickle
from model_params_sp import ModelParamSpace
from utils import logging_utils
from utils import time_utils

import cofig



class Learner:
    def __init__(self, learner_name, param_dict):
        self.learner_name = learner_name
        self.param_dict = param_dict
        self.learner = self._get_learner()

    def __str__(self):
        return self.learner_name

    def _get_learner(self):
        # xgboost
        if self.learner_name in ["clf_xgb_linear", "clf_xgb_tree"]:
            return XGBClassifier(**self.param_dict)
        # sklearn
        if self.learner_name == "clf_skl_random_logistic":
            return LogisticRegression(**self.param_dict)
        if self.learner_name == "clf_skl_svC":
            return SVC(**self.param_dict)
        if self.learner_name == "clf_skl_lsvr":
            return LinearSVC(**self.param_dict)
        if self.learner_name == "clf_skl_etr":
            return ExtraTreesClassifier(**self.param_dict)
        if self.learner_name == "clf_skl_adaboost":
            return AdaBoostClassifier(**self.param_dict)

    def fit(self, X, y, feature_names=None):
        if feature_names is not None:
            self.learner.fit(X, y, feature_names)
        else:
            self.learner.fit(X, y)
        return self

    def predict(self, X, feature_names=None):
        if feature_names is not None:
            y_pred = self.learner.predict(X, feature_names)
        else:
            y_pred = self.learner.predict(X)
        # relevance is in [1,3]
        y_pred = np.clip(y_pred, 1., 3.)
        return y_pred


class Task:
    def __init__(self, learner, logger):
        self.learner = learner
        self.logger = logger
        self.n_iters = cofig.N_FOLDS
        self.score_cv_mean = 0
        self.score_cv_std = 0


    def load_data(self):
        train = pickle._load(cofig.FINISHDATA_TRAIN)
        test = pickle._load(cofig.FINISHDATA_TEST)
        target = pickle._load(cofig.FINISHDATA_TARGET)
        return train, test, target

    def __str__(self):
        return "[Learner@%s]" % (str(self.learner))

    def _get_splitter(self):
        return pickle._load(cofig.SPLITTER)

    def get_score(self, ypred, ytrue):
        return roc_auc_score(ypred, ytrue)

    def cv(self):
        start = time.time()
        self.logger.info("=" * 50)
        self.logger.info("Task")
        self.logger.info("      %s" % str(self.__str__()))
        self.logger.info("Param")
        self._print_param_dict(self.learner.param_dict)
        self.logger.info("Result")
        self.logger.info("      Run      AUC       ")

        train, test, target = self.load_data()

        splitter = self._get_splitter()
        score_cv = np.zeros(shape=(1, self.n_iters))
        for i, (train_idx, valid_idx) in enumerate(splitter):
            train_ar, valid_ar = train[train_idx, :], train[valid_idx, :]
            self.learner.fit(train_ar, target[train_idx])
            yvalid = self.learner.predict(valid_ar)
            score = self.get_score(yvalid, target[valid_idx])
            self.logger("     %d       %.6f       "%(i, score))
            score_cv[0, i] = score

        self.score_cv_mean = np.mean(score_cv, axis=1)
        self.score_cv_std = np.std(score_cv, axis=1)

        end = time.time()
        sec = start - end
        self.logger("      score_mean:%.6f     "%self.score_cv_mean)
        self.logger("      score_std:%.f      "%self.score_cv_std)
        self.logger("      sec:%.6f           "%sec)
        self.logger.info("-" * 50)
        return self

    def refit(self):
        train, test, target = self.load_data()

        self.learner.fit(train, target)
        ypred = self.learner.predict(test)

        submission = pd.DataFrame(ypred, columns='%s_pred'%self.__str__())

        fname = "%s/%s/test.pred.%s.csv"%(cofig.OUTPUT_DIR, "All", self.__str__())
        pickle.save(fname, submission)

        return self

    def go(self):
        self.cv()
        self.refit()

class TaskOptimizer:
    def __init__(self, task_mode, learner_name, logger,
                    max_evals=100, verbose=True, refit_once=False, plot_importance=False):
        self.task_mode = task_mode
        self.learner_name = learner_name
        self.logger = logger
        self.max_evals = max_evals
        self.verbose = verbose
        self.refit_once = refit_once
        self.plot_importance = plot_importance
        self.trial_counter = 0
        self.model_param_space = ModelParamSpace(self.learner_name)

    def _obj(self, param_dict):
        self.trial_counter += 1
        param_dict = self.model_param_space._convert_int_param(param_dict)
        learner = Learner(self.learner_name, param_dict)
        self.task = Task(learner, self.logger)
        self.task.go()
        ret = {
            'loss':-self.task.score_cv_mean,
            'attachments':{
                'std': self.task.score_cv_std,
            },
            'stats':STATUS_OK
        }
        return ret

    def run(self):
        start = time.time()
        trials = Trials()
        best = min(self._obj, self.model_param_space.build_space(), tpe.suggest, self.max_evals, trials)
        best_params = space_eval(self.model_param_space.build_space(), best)
        best_params = self.model_param_space._convert_int_param(best_params)
        trial_score = np.asarray(trials.losses(), dtype=float)
        best_ind = np.argmin(trial_score)
        best_score_mean = trial_score[best_ind]
        best_score_std = trials.trial_attachments(trials.trials[best_ind])["std"]
        self.logger.info("-" * 50)
        self.logger.info("Best Score")
        self.logger.info("      Mean: %.6f" % -best_score_mean)
        self.logger.info("      std: %.6f" % best_score_std)
        self.logger.info("Best param")
        self.task._print_param_dict(best_params)
        end = time.time()
        _sec = end - start
        _min = int(_sec / 60.)
        self.logger.info("Time")
        self.logger.info('-'* 50)







def main(options):
    logname = "[Feat@%s]_[Learner@%s]_hyperopt_%s.log"%(
        options.feature_name, options.learner_name, time_utils._timestamp())
    logger = logging_utils._get_logger(cofig.LOG_DIR, logname)
    optimizer = TaskOptimizer(options.task_mode, options.learner_name,
        logger, options.max_evals, verbose=True,
        refit_once=options.refit_once, plot_importance=options.plot_importance)
    optimizer.run()

def parse_args(parser):
    parser.add_option("-m", "--mode", type="string", dest="task_mode",
        help="task mode", default="single")
    parser.add_option("-f", "--feat", type="string", dest="feature_name",
        help="feature name", default="basic")
    parser.add_option("-l", "--learner", type="string", dest="learner_name",
        help="learner name", default="reg_skl_ridge")
    parser.add_option("-e", "--eval", type="int", dest="max_evals",
        help="maximun number of evals for hyperopt", default=100)
    parser.add_option("-o", default=False, action="store_true", dest="refit_once",
        help="stacking refit_once")
    parser.add_option("-p", default=False, action="store_true", dest="plot_importance",
        help="plot feautre importance (currently only for xgboost)")

    (options, args) = parser.parse_args()
    return options, args


if __name__ == "__main__":

    parser = OptionParser()
    options, args = parse_args(parser)
    main(options)


