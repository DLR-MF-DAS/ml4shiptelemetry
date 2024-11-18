from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.multioutput import MultiOutputClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from typing import List
import numpy as np
from functools import partial

def print_performance(target_names, title, score_dict, num_decimals: int = 4):
        
    print(title)
    if target_names is None:
        for score_name, score in score_dict.items():
            print(f'{score_name}: {", ".join([f"{s:.{num_decimals}f}" for s in score])}')
    else:
        metric_strs = [[f'{s:.{num_decimals}f}' for s in score] for score in score_dict.values()]
        # Calculate number of characters needed for printing an aligned table
        metric_name_len = max([len(m) for m in score_dict.keys()])
        max_length = max(max([len(s) for s in target_names]), num_decimals+2) + 2
        length_str = f':>{max_length}'
        row_format = f"{{:>{metric_name_len}}}" + f"{{{length_str}}}" * (len(target_names))
        # Print
        print(row_format.format("", *target_names))
        for metric, row in zip(score_dict.keys(), metric_strs):
            print(row_format.format(metric, *row))



class RFRegressor:
    
    # Define metric functions, possibly with partially prefilled values
    score_r2 = partial(r2_score, multioutput='raw_values')
    score_mae = partial(mean_absolute_error, multioutput='raw_values')

    regressor = None

    def __init__(self, num_trees, maximum_depth, seed = 18, target_names: List[str] = None, scores=None):
        self.num_trees = num_trees
        self.maximum_depth = maximum_depth
        self.seed = seed
        self.regressor = RandomForestRegressor(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed, n_jobs=-1)
        self.target_names = target_names
        if scores is not None:
            self.scores = scores
        else:
            self.scores = {'R2': 'r2', 'MAE': 'mae'}


    def train(self, x, y):
        self.regressor.fit(x, y)


    def crossvalidate(self, x, y, num_splits = 5, time_series=False, shuffle=True):
        if time_series:
            kf = TimeSeriesSplit(n_splits=num_splits)
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle)

        for iteration, (train, test) in enumerate(kf.split(x)):
            # Fit regressor
            regressor = RFRegressor(self.num_trees, maximum_depth=self.maximum_depth, target_names=self.target_names, scores=self.scores)
            regressor.train(x[train], y[train])
            # Predict
            train_score = regressor.calculate_all_scores(x[train], y[train])
            test_score = regressor.calculate_all_scores(x[test], y[test])
            # Print performance
            self.print_performance(title=f"Iteration {iteration} - Training Score:", score_dict=train_score, num_decimals=4)
            self.print_performance(title=f"Iteration {iteration} - Test Score:", score_dict=test_score, num_decimals=4)
            print("----------------")


    def predict(self, x):
        return self.regressor.predict(x)
    

    def score(self, x, y_true, metric: str):
        pred = self.predict(x)
        if metric == 'r2':
            metric_fun = self.score_r2
        elif metric == 'mae':
            metric_fun = self.score_mae
        else:
            raise ValueError(f"Metric {metric} is not implemented. Currently available metrics are 'r2' for R2 score, 'mae' for mean absolute error.")
        return metric_fun(y_true, pred)


    def calculate_all_scores(self, x, y):
        """Calculate all scores for all targets"""
        scores = {}
        for metric, metric_fun in self.scores.items():
            scores[metric] = self.score(x, y, metric_fun)
        return scores
    
    def print_performance(self, title, score_dict, num_decimals: int = 4):
        print_performance(self.target_names, title, score_dict, num_decimals)


class RFClassifier:
    
    # Define metric functions, possibly with partially prefilled values
    score_acc = partial(accuracy_score)
    score_confusion = partial(confusion_matrix)
    score_f1 = partial(f1_score, average='micro')
    score_balacc = partial(balanced_accuracy_score)

    classifier = None

    def __init__(self, num_trees, maximum_depth, seed = 18, target_names: List[str] = None, scores=None, model='rf'):
        self.num_trees = num_trees
        self.maximum_depth = maximum_depth
        self.seed = seed
        if model == 'rf':
            self.classifier = RandomForestClassifier(n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, random_state = self.seed, class_weight='balanced_subsample', n_jobs=-1)
        elif model == 'balanced_rf':
            self.classifier = MultiOutputClassifier(BalancedRandomForestClassifier(
                n_estimators = self.num_trees, max_features = 'sqrt', max_depth = self.maximum_depth, 
                random_state = self.seed, n_jobs=-1, #class_weight='balanced_subsample',
                sampling_strategy='all', replacement=True, bootstrap=False))
        self.n_targets = None
        self.target_names = target_names

        if scores is not None:
            self.scores = scores
        else:
            self.scores = {'Accuracy': 'acc', 'Balanced Acc.': 'balacc'}


    def train(self, x, y):
        self.set_n_targets(y)
        self.classifier.fit(x, y)


    def crossvalidate(self, x, y, num_splits = 5, time_series=False, shuffle=True):
        if time_series:
            kf = TimeSeriesSplit(n_splits=num_splits)
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle)

        for iteration, (train, test) in enumerate(kf.split(x)):
            # Fit classifier
            classifier = RFClassifier(self.num_trees, maximum_depth=self.maximum_depth, target_names=self.target_names, scores=self.scores)
            classifier.train(x[train, :], y[train, :])
            # Predict
            train_score = classifier.calculate_all_scores(x[train], y[train])
            test_score = classifier.calculate_all_scores(x[test], y[test])
            # Print performance
            self.print_performance(title=f"Iteration {iteration} - Training Score:", score_dict=train_score, num_decimals=4)
            self.print_performance(title=f"Iteration {iteration} - Test Score:", score_dict=test_score, num_decimals=4)
            print("----------------")


    def predict(self, x):
        return self.classifier.predict(x)


    def predict_proba(self, x):
        return self.classifier.predict_proba(x)
    

    def set_n_targets(self, y):
        if len(y.shape) > 1:
            self.n_targets = y.shape[1]
        else:
            self.n_targets = 1
    

    def score(self, x, y_true, metric: str):
        if self.n_targets is None:
            self.set_n_targets(y_true)

        # Configure the metric functions and what prediction to use, labels or probabilities
        if metric == 'f1':
            metric_fun = self.score_f1
        elif metric == 'acc':
            metric_fun = self.score_acc
        elif metric == 'balacc':
            metric_fun = self.score_balacc
        else:
            raise ValueError(f"Metric {metric} is not implemented. Currently available metrics are 'acc' for accuracy, 'balacc' for balanced accuracy, and 'f1' for F1 score.")
        
        # Predict probabilities and labels
        pred = self.predict(x)
        
        # Calculate integer scores
        scores = [float(metric_fun(y_true[:, itarget], pred[:, itarget])) for itarget in np.arange(self.n_targets)]
        return scores
    

    def confusion_matrix(self, x, y_true, normalize=None, print_matrix=False):
        pred = self.predict(x)
        cms = [self.score_confusion(y_true[:, itarget], pred[:, itarget], normalize=normalize, labels=self.classifier.classes_[itarget])
                for itarget in np.arange(self.n_targets)]
        print('Confusion matrix')
        if print_matrix:
            for ind, cm in enumerate(cms):
                classes = self.classifier.classes_[ind]
                label_length = max([len(str(c)) for c in self.classifier.classes_[ind]] + [len('Flag')])
                val_length = np.maximum(len(str(np.max(cm)))+2, label_length)
                length_str = f':>{val_length}'
                row_format = f"{{:>{label_length}}}" + f"{{{length_str}}}" * (len(classes))
                print('--------')
                if self.target_names is None:
                    print(f'Target {ind}')
                else:    
                    print(f'Target: {self.target_names[ind]}')
                print(row_format.format("", ' Prediction', *[""]*(len(classes)-1)))
                print(row_format.format("Flag", *classes))
                for metric, row in zip(classes, cm):
                    print(row_format.format(metric, *row))
        else:
            return cms


    def calculate_all_scores(self, x, y):
        """Calculate all scores for all targets"""
        scores = {}
        for metric, metric_fun in self.scores.items():
            scores[metric] = self.score(x, y, metric_fun)
        return scores
    

    def print_performance(self, title, score_dict, num_decimals: int = 4):
        print_performance(self.target_names, title, score_dict, num_decimals)
