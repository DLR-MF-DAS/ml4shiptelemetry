from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, TimeSeriesSplit
from typing import List
import numpy as np


class RFRegressor:
    
    regressor = None

    def __init__(self, num_trees, maximum_depth, seed = 18, target_names: List[str] = None, scores=None):
        self.num_trees = num_trees
        self.maximum_depth = maximum_depth
        self.seed = seed
        self.regressor = RandomForestRegressor(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed)
        self.target_names = target_names
        if scores is not None:
            self.scores = scores
        else:
            self.scores = {'R2': r2_score, 'MAE': mean_absolute_error}


    def train(self, x, y):
        self.regressor.fit(x, y)


    def crossvalidate(self, x, y, num_splits = 5, time_series=False, shuffle=True):
        if time_series:
            kf = TimeSeriesSplit(n_splits=num_splits)
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle)

        for iteration, (train, test) in enumerate(kf.split(x)):
            regressor = RandomForestRegressor(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed).fit(x[train], y[train])
            prediction_train = regressor.predict(x[train])
            prediction_test = regressor.predict(x[test])

            # Print scores
            train_scores = []
            test_scores = []
            for metric in self.scores.values():
                train_scores.append(metric(y[train], prediction_train, multioutput='raw_values'))
                test_scores.append(metric(y[test], prediction_test, multioutput='raw_values'))
            self.print_performance( title=f"Iteration {iteration} - Training Score:", scores=train_scores, score_names=self.scores.keys(), num_decimals=4)
            self.print_performance( title=f"Iteration {iteration} - Test Score:", scores=test_scores, score_names=self.scores.keys(), num_decimals=4)
            print("----------------")


    def predict(self, x):
        return self.regressor.predict(x)
    

    def print_performance(self, title, scores: List[float], score_names: List[str], num_decimals: int = 4):
        
        print(title)
        if self.target_names is None:
            for score, score_name in zip(scores, score_names):
                print(f'{score_name}: {", ".join([f"{s:.{num_decimals}f}" for s in score])}')
        else:
            max_length = max(max([len(s) for s in self.target_names]), num_decimals+2) + 2
            metric_strs = [[f'{s:.{num_decimals}f}' for s in score] for score in scores]
            length_str = f':>{max_length}'
            row_format =f"{{{length_str}}}" * (len(self.target_names) + 1)
            print(row_format.format("", *self.target_names))
            for metric, row in zip(score_names, metric_strs):
                print(row_format.format(metric, *row))  