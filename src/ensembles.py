import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class MyRandomForest:

    def __init__(self, n_estimators= 100, k_frac= 0.25, max_depth= 10):
        self.n_estimators = n_estimators
        self.k_frac = 0.25
        self.max_depth = max_depth

    def fit(self, X, y):
        self.models = []
        self.inds = []
        self.objs = []
        for i in range(self.n_estimators):
            self.models.append(DecisionTreeRegressor(max_depth= self.max_depth))
            self.inds.append(np.random.choice(np.arange(X.shape[1]), round(X.shape[1] * self.k_frac), replace= False))
            self.objs.append(np.random.choice(np.arange(X.shape[0]), round(X.shape[0] * 0.6))) # 0.6 - оптимальный параметр из лекции
            self.models[-1].fit(X.drop(self.inds[-1], axis= 1).iloc[self.objs[-1], :], y[self.objs[-1]])

    def predict(self, X):
        pred = np.zeros(X.shape[0])
        for i in range(self.n_estimators):
            X_bagged = X.drop(self.inds[i], axis= 1)
            pred += self.models[i].predict(X_bagged)
        return pred / self.n_estimators

class MyGradBoost:

    def __init__(self, n_estimators= 100, k_frac= 0.25, max_depth= 10, learning_rate= 0.1):
        self.n_estimators = n_estimators
        self.k_frac = k_frac
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.models = []
        pred = []
        # init first predictor
        self.models.append(DecisionTreeRegressor(max_depth= self.max_depth))
        self.models[-1].fit(X, y)
        pred = self.models[-1].predict(X)
        for t in range(1, self.n_estimators):
            self.models.append(DecisionTreeRegressor(max_depth= self.max_depth))
            self.models[-1].fit(X, y - pred)
            pred += self.learning_rate * self.models[-1].predict(X)

    def predict(self, X):
        pred = self.models[0].predict(X)
        for t in range(1, self.n_estimators):
            pred += self.learning_rate * self.models[t].predict(X)
        return pred

