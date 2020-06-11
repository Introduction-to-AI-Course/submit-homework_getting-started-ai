#-*- coding utf8 -*-
"""
@autho:
@brief: model parameter space
"""

import numpy as np
import hyperopt as hp

import cofig
## xgboost
xgb_random_seed = cofig.RANDOM_SEED
xgb_nthread = cofig.NUM_CORES
xgb_n_estimators_min = 100
xgb_n_estimators_max = 1000
xgb_n_estimators_step = 10

## sklearn
skl_random_seed = cofig.RANDOM_SEED
skl_n_jobs = cofig.NUM_CORES
skl_n_estimators_min = 100
skl_n_estimators_max = 1000
skl_n_estimators_step = 10

# ---------------------------- XGBoost ---------------------------------------
## classifier with linear booster
param_space_xgb_linear = {
    "booster": "gblinear",
    "objective": "reg:linear",
    "base_score": cofig.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "reg_lambda_bias" : hp.quniform("reg_lambda_bias", 0, 3, 0.1),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}

## classifer with tree booster
param_space_xgb_tree = {
    "booster": "gbtree",
    "objective": "reg:linear",
    "base_score": cofig.BASE_SCORE,
    "n_estimators" : hp.quniform("n_estimators", xgb_n_estimators_min, xgb_n_estimators_max, xgb_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "gamma": hp.loguniform("gamma", np.log(1e-10), np.log(1e1)),
    "reg_alpha" : hp.loguniform("reg_alpha", np.log(1e-10), np.log(1e1)),
    "reg_lambda" : hp.loguniform("reg_lambda", np.log(1e-10), np.log(1e1)),
    "min_child_weight": hp.loguniform("min_child_weight", np.log(1e-10), np.log(1e2)),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "subsample": hp.quniform("subsample", 0.1, 1, 0.05),
    "colsample_bytree": 1,
    "colsample_bylevel": hp.quniform("colsample_bylevel", 0.1, 1, 0.05),
    "nthread": xgb_nthread,
    "seed": xgb_random_seed,
}


param_space_clf_skl_random_logistic = {
    "alpha": hp.loguniform("alpha", np.log(0.01), np.log(20)),
    "normalize": hp.choice("normalize", [True, False]),
    "poly": hp.choice("poly", [False]),
    "n_estimators": hp.quniform("n_estimators", 2, 50, 2),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
    "random_state": skl_random_seed
}

## linear support vector classifier
param_space_clf_skl_lsvc = {
    "normalize": hp.choice("normalize", [True, False]),
    "C": hp.loguniform("C", np.log(1), np.log(100)),
    "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    "loss": hp.choice("loss", ["epsilon_insensitive", "squared_epsilon_insensitive"]),
    "random_state": skl_random_seed,
}

## support vector classifier
param_space_clf_skl_svc = {
    "normalize": hp.choice("normalize", [True]),
    "C": hp.loguniform("C", np.log(1), np.log(1)),
    "gamma": hp.loguniform("gamma", np.log(0.001), np.log(0.1)),
    "degree": hp.quniform("degree", 1, 3, 1),
    "epsilon": hp.loguniform("epsilon", np.log(0.001), np.log(0.1)),
    "kernel": hp.choice("kernel", ["rbf", "poly"])
}

## extra trees classifierr
param_space_clf_skl_etr = {
    "n_estimators": hp.quniform("skl_etr__n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "max_features": hp.quniform("skl_etr__max_features", 0.1, 1, 0.05),
    "min_samples_split": hp.quniform("skl_etr__min_samples_split", 1, 15, 1),
    "min_samples_leaf": hp.quniform("skl_etr__min_samples_leaf", 1, 15, 1),
    "max_depth": hp.quniform("skl_etr__max_depth", 1, 10, 1),
    "random_state": skl_random_seed,
    "n_jobs": skl_n_jobs,
    "verbose": 0,
}


## adaboost classifier
param_space_clf_skl_adaboost = {
    "base_estimator": hp.choice("base_estimator", ["dtr", "etr"]),
    "n_estimators": hp.quniform("n_estimators", skl_n_estimators_min, skl_n_estimators_max, skl_n_estimators_step),
    "learning_rate" : hp.qloguniform("learning_rate", np.log(0.002), np.log(0.1), 0.002),
    "max_features": hp.quniform("max_features", 0.1, 1, 0.05),
    "max_depth": hp.quniform("max_depth", 1, 10, 1),
    "loss": hp.choice("loss", ["linear", "square", "exponential"]),
    "random_state": skl_random_seed,
}


# -------------------------------------- All ---------------------------------------------
param_space_dict = {
    # xgboost
    "clf_xgb_tree": param_space_xgb_tree,
    "clf_xgb_linear": param_space_xgb_linear,
    # sklearn
    "clf_skl_random_logistic": param_space_clf_skl_random_logistic,
    "clf_skl_lsvc": param_space_clf_skl_lsvc,
    "clf_skl_svc": param_space_clf_skl_svc,
    "clf_skl_etr": param_space_clf_skl_etr,
    "clf_skl_adaboost": param_space_clf_skl_adaboost,
}

int_params = [
    "num_round", "n_estimators", "min_samples_split", "min_samples_leaf",
    "n_neighbors", "leaf_size", "seed", "random_state", "max_depth", "degree",
    "hidden_units", "hidden_layers", "batch_size", "nb_epoch", "dim", "iter",
    "factor", "iteration", "n_jobs", "max_leaf_forest", "num_iteration_opt",
    "num_tree_search", "min_pop", "opt_interval",
]
int_params = set(int_params)


class ModelParamSpace:
    def __init__(self, learner_name):
        s = "Wrong learner_name, " + \
            "see model_param_space.py for all available learners."
        assert learner_name in param_space_dict, s
        self.learner_name = learner_name

    def _build_space(self):
        return param_space_dict[self.learner_name]

    def _convert_int_param(self, param_dict):
        if isinstance(param_dict, dict):
            for k,v in param_dict.items():
                if k in int_params:
                    param_dict[k] = int(v)
                elif isinstance(v, list) or isinstance(v, tuple):
                    for i in range(len(v)):
                        self._convert_int_param(v[i])
                elif isinstance(v, dict):
                    self._convert_int_param(v)
        return param_dict
