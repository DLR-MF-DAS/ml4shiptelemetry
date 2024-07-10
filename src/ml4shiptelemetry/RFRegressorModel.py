import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


class RFRegressor:
    
    regressor = None

    def __init__(self, num_trees, maximum_depth, seed = 18):
        self.num_trees = num_trees
        self.maximum_depth = maximum_depth
        self.seed = seed
        self.regressor = RandomForestRegressor(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed)

    def train(self, x, y):
        self.regressor.fit(x, y)

    def crossvalidate(self, x, y, num_splits = 5):
        kf = KFold(n_splits=num_splits, shuffle=True)
        iteration = 1
        for train, test in kf.split(x):
            regressor = RandomForestRegressor(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed).fit(x[train], y[train])
            prediction = regressor.predict(x[train])
            print(f"Iteration {iteration} - Training R2 Score:")
            print(r2_score(y[train], prediction, multioutput='raw_values'))
            prediction = regressor.predict(x[test])
            print(f"Iteration {iteration} - Test R2 Score:")
            print(r2_score(y[test], prediction, multioutput='raw_values'))
            print("----------------")
            iteration += 1

    def predict(self, x):
        return self.regressor.predict(x)