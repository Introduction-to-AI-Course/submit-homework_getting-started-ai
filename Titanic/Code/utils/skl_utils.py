# -*- coding utf8 -*-\
"""
@author:
@brief: some models for learner
"""

import numpy as np
import sklearn.svm
import sklearn.neighbors
import sklearn.ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

class SVC:
    def __init__(self, kernel='rbf', degree=3, gamma='auto', C=1.0,
                 epsilon=0.1, normalize=True):
        svr = sklearn.svm.SVC(kernel=kernel, degree=degree,
                              gamma=gamma, C=C, epsilon=epsilon)
        if normalize:
            self.model = Pipeline([('ss', StandardScaler()), ('svr', svr)])
        else:
            self.model = svr

    def __str__(self):
        return "svm_SVC"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred


class LinearSVC:
    def __init__(self, epsilon=0.0, C=1.0, loss='epsilon_insensitive',
                 random_state=None, normalize=True):
        lsvr = sklearn.svm.LinearSVC(epsilon=epsilon, C=C,
                                     loss=loss, random_state=random_state)
        if normalize:
            self.model = Pipeline([('ss', StandardScaler()), ('lsvr', lsvr)])
        else:
            self.model = lsvr

    def __str__(self):
        return "LinearSVC"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

class AdaBoostClassifier:
    def __init__(self, base_estimator=None, n_estimators=50, max_features=1.0,
                max_depth=6, learning_rate=1.0, loss='linear', random_state=None):
        if base_estimator and base_estimator == 'etr':
            base_estimator = ExtraTreeClassifier(max_depth=max_depth,
                                        max_features=max_features)
        else:
            base_estimator = DecisionTreeClassifier(max_depth=max_depth,
                                        max_features=max_features)

        self.model = sklearn.ensemble.AdaBoostClassifier(
                                    base_estimator=base_estimator,
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    random_state=random_state,
                                    loss=loss)

    def __str__(self):
        return "AdaBoostClassifier"

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

class RandomLogistc:
    def __init__(self, alpha=1.0, normalize=True, poly=False,
                    n_estimators=10, max_features=1.0,
                    bootstrap=True, subsample=1.0,
                    random_state=2016):
        self.alpha = alpha
        self.normalize = normalize
        self.poly = poly
        self.n_estimators = n_estimators
        if isinstance(max_features, float):
            assert max_features > 0 and max_features <= 1
        self.max_features = max_features
        self.bootstrap = bootstrap
        assert subsample > 0 and subsample <= 1
        self.subsample = subsample
        self.random_state = random_state
        self.lr_list = [0]*self.n_estimators
        self.feature_idx_list = [0]*self.n_estimators

    def __str__(self):
        return "Logistic"

    def _random_feature_idx(self, fdim, random_state):
        rng = np.random.RandomState(random_state)
        if isinstance(self.max_features, int):
            size = min(fdim, self.max_features)
        else:
            size = int(fdim * self.max_features)
        idx = rng.permutation(fdim)[:size]
        return idx

    def _random_sample_idx(self, sdim, random_state):
        rng = np.random.RandomState(random_state)
        size = int(sdim * self.subsample)
        if self.bootstrap:
            idx = rng.randint(sdim, size=size)
        else:
            idx = rng.permutation(sdim)[:size]
        return idx

    def fit(self, X, y):
        sdim, fdim = X.shape
        for i in range(self.n_estimators):
            lr = LogisticRegression(alpha=self.alpha, normalize=self.normalize, random_state=self.random_state)
            fidx = self._random_feature_idx(fdim, self.random_state+i*100)
            sidx = self._random_sample_idx(sdim, self.random_state+i*10)
            X_tmp = X[sidx][:,fidx]
            if self.poly:
                X_tmp = PolynomialFeatures(degree=2).fit_transform(X_tmp)[:,1:]
            lr.fit(X_tmp, y[sidx])
            self.lr_list[i] = lr
            self.feature_idx_list[i] = fidx
        return self

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
            fidx = self.feature_idx_list[i]
            lr = self.lr_list[i]
            X_tmp = X[:,fidx]
            if self.poly:
                X_tmp = PolynomialFeatures(degree=2).fit_transform(X_tmp)[:,1:]
            y_pred[:,i] = lr.predict(X_tmp)
        y_pred = np.mean(y_pred, axis=1)
        return y_pred
