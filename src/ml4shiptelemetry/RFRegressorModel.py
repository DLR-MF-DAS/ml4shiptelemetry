from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix, f1_score, balanced_accuracy_score
from sklearn.model_selection import KFold, TimeSeriesSplit, GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from typing import List, Dict
import numpy as np
from functools import partial
import time
import logging


def print_performance(target_names: List[str], title: str, score_dict: Dict[str, float], num_decimals: int = 4):
    """Prints the performance measured in one or multiple metrics of a single model

    Parameters
    ----------
    target_names : List[str]
        Name(s) of the target(s)
    title : str
        Title of the performance chart
    score_dict : Dict[str, float]
        Dictionary containing the name of the score and its value for every target
    num_decimals : int
        Number of decimals to plot
    """
    logger = logging.getLogger()
    message = title
    if target_names is None:
        for score_name, score in score_dict.items():
            message += '\n' +  f'{score_name}: {", ".join([f"{s:.{num_decimals}f}" for s in score])}'
    else:
        metric_strs = [[f'{s:.{num_decimals}f}' for s in score] for score in score_dict.values()]
        # Calculate number of characters needed for printing an aligned table
        metric_name_len = max([len(m) for m in score_dict.keys()])
        max_length = max(max([len(s) for s in target_names]), num_decimals+2) + 2
        length_str = f':>{max_length}'
        row_format = f"{{:>{metric_name_len}}}" + f"{{{length_str}}}" * (len(target_names))
        message += '\n' +  row_format.format("", *target_names)
        for metric, row in zip(score_dict.keys(), metric_strs):
            message += '\n' +  row_format.format(metric, *row)
    logger.info(message)


class RFRegressor:
    
    # Define metric functions, possibly with partially prefilled values
    score_r2 = staticmethod(partial(r2_score, multioutput='raw_values'))
    score_mae = staticmethod(partial(mean_absolute_error, multioutput='raw_values'))

    regressor = None

    def __init__(self, target_names: List[str] = None, scores=None, **kwargs):
        self.regressor = RandomForestRegressor(**kwargs)
        self.target_names = target_names
        self.logger = logging.getLogger()
        if scores is not None:
            self.scores_dict = scores
        else:
            self.scores_dict = {'R2': 'r2', 'MAE': 'mae'}


    def fit(self, x, y):
        self.regressor.fit(x, y)


    def crossvalidate(self, x, y, cv_params=None, num_splits = 5, time_series = False, 
                      shuffle = True, verbose=1, n_jobs = 1):
        """Perform cross-validation hyperparameter tuning

        Parameters
        ----------
        x : numpy.array
            Data to perform cross-validation on.
        y : numpy.array
            Targets for each row in x.
        num_splits : int
            Number of Splits in the cross-validation. Default is 5.
        time_series : bool
            If True, use sklearn.TimeSeriesSplit. Otherwise use sklearn.KFold. Default is False.
        shuffle : bool
            Shuffle data in KFold. Not used in case of time_series=True. Recommended to keep False. Default is False.
        verbose : int
            Verbosity level of sklearn.GridSearchCV. Default is 1.
        n_jobs : int
            Number of concurrent parameter combinations. Default is 1.
        
        Returns
        -------
        sklearn.RandomForestRegressor
        """
        t_start = time.time()
        if time_series:
            kf = TimeSeriesSplit(n_splits=num_splits)
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle)

        # Perform Randomized hyperparameter search
        # Parameters to search
        if cv_params is None:
            cv_params = {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_features': ['sqrt'],
                'criterion': ['friedman_mse'],
                'max_depth': [None]
            }
        
        param_search = GridSearchCV(
            estimator=self.regressor, 
            param_grid=cv_params,
            cv=kf,              # Number of cross-validation folds
            verbose=verbose,         # Show progress during search
            n_jobs=n_jobs,          # Use all processors for parallel processing
            refit=True          # Retrain model with best parameters on full dataset
        )
        param_search.fit(x, y)
        best_params = param_search.best_params_
        params_str = ", ".join([f"{s} {best_params[s]}" for s in cv_params.keys()])
        message = f"Best model parameters: {params_str}"
        message += '\n' +  f"R2 of best model from CV: {param_search.best_score_:.3f}."
        message += '\n' +  f"Time for GridSearchCV iterations: {time.time() - t_start:.1f}s"
        self.logger.info(message)
        # Overwrite regressor
        self.regressor = param_search.best_estimator_


    def set_params(self, **params):
        return self.regressor.set_params(**params)
    

    def get_params(self, deep=True):
        return self.regressor.get_params(deep)

    def predict(self, x):
        return self.regressor.predict(x)
    

    def score(self, X, y, sample_weight=None):
        return self.regressor.score(X, y, sample_weight)
    

    def calc_score(self, x, y_true, metric: str):
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
        for metric, metric_fun in self.scores_dict.items():
            scores[metric] = self.calc_score(x, y, metric_fun)
        return scores
    
    def print_performance(self, title, score_dict, num_decimals: int = 4):
        print_performance(self.target_names, title, score_dict, num_decimals)


class RFClassifier:
    
    # Define metric functions, possibly with partially prefilled values
    score_acc = staticmethod(partial(accuracy_score))
    score_confusion = staticmethod(partial(confusion_matrix))
    score_f1 = staticmethod(partial(f1_score, average='micro'))
    score_balacc = staticmethod(partial(balanced_accuracy_score))

    classifier = None

    def __init__(self, target_names: List[str] = None, scores_dict=None, **kwargs):
        self.classifier = BalancedRandomForestClassifier(**kwargs)
        self.n_targets = None
        self.target_names = target_names
        self.logger = logging.getLogger()
        if scores_dict is not None:
            self.scores_dict = scores_dict
        else:
            self.scores_dict = {'Accuracy': 'acc', 'Balanced Acc.': 'balacc'}


    def fit(self, x, y):
        self.set_n_targets(y)
        self.classifier.fit(x, y)


    def crossvalidate(self, x, y, cv_params, num_splits = 5, time_series=False, 
                      shuffle=True, verbose=1, n_jobs=1):
        """Perform cross-validation hyperparameter tuning

        Parameters
        ----------
        x : numpy.array
            Data to perform cross-validation on.
        y : numpy.array
            Targets for each row in x.
        num_splits : int
            Number of Splits in the cross-validation. Default is 5.
        time_series : bool
            If True, use sklearn.TimeSeriesSplit. Otherwise use sklearn.KFold. Default is False.
        shuffle : bool
            Shuffle data in KFold. Not used in case of time_series=True. Recommended to keep False. Default is False.
        verbose : int
            Verbosity level of sklearn.GridSearchCV. Default is 1.
        n_jobs : int
            Number of concurrent parameter combinations. Default is 1.
        
        Returns
        -------
        imblearn.ensemble.BalancedRandomForestClassifier
        """
        
        t_start = time.time()
        if time_series:
            kf = TimeSeriesSplit(n_splits=num_splits)
        else:
            kf = KFold(n_splits=num_splits, shuffle=shuffle)

        # Parameters to seach
        if cv_params is None:
            cv_params = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_features': ['sqrt'],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None]
        }

        param_search = GridSearchCV(
            estimator=self.classifier, 
            param_grid=cv_params,
            cv=kf,              # Number of cross-validation folds
            verbose=verbose,         # Show progress during search
            n_jobs=n_jobs,          # Use all processors for parallel processing
            refit=True          # Retrain model with best parameters on full dataset
        )

        param_search.fit(x, y)
        best_params = param_search.best_params_
        params_str = ", ".join([f"{s}' {best_params[s]}" for s in cv_params.keys()])
        message = f"Best model parameters: {params_str}"
        message += '\n' +  f"Mean balanced accuracy of best model from CV: {100*param_search.best_score_:.1f}%."
        message += '\n' +  f"Time for GridSearchCV iterations: {time.time() - t_start:.1f}s"
        self.logger.info(message)
        # Overwrite classifier
        self.classifier = param_search.best_estimator_


    def set_params(self, **params):
        self.classifier.set_params(**params)
    

    def get_params(self, deep=True):
        return self.classifier.get_params(deep)#params
    

    def predict(self, x):
        return self.classifier.predict(x)


    def predict_proba(self, x):
        return self.classifier.predict_proba(x)
    

    def set_n_targets(self, y):
        if len(y.shape) > 1:
            self.n_targets = y.shape[1]
        else:
            self.n_targets = 1

    
    def score(self, X, y, sample_weight=None):
        return self.classifier.score(X, y, sample_weight)
    

    def calc_score(self, x, y_true, metric: str):
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
        if self.n_targets > 1:
            scores = [float(metric_fun(y_true[:, itarget], pred[:, itarget])) for itarget in np.arange(self.n_targets)]
        else:
            scores = [float(metric_fun(y_true, pred))]
        return scores
    

    def confusion_matrix(self, x, y_true, normalize=None, print_matrix=False):
        pred = self.predict(x)
        if self.n_targets > 1:
            cms = [self.score_confusion(y_true[:, itarget], pred[:, itarget], normalize=normalize, labels=self.classifier.classes_[itarget])
                    for itarget in np.arange(self.n_targets)]
        else:
            cms = cms = [self.score_confusion(y_true, pred, normalize=normalize, labels=self.classifier.classes_)]
        message = '\nConfusion matrix'
        if print_matrix:
            for ind, cm in enumerate(cms):
                if self.n_targets > 1:
                    classes = self.classifier.classes_[ind]
                    label_length = max([len(str(c)) for c in self.classifier.classes_[ind]] + [len('Flag')])
                else:
                    classes = self.classifier.classes_
                    label_length = max([len(str(c)) for c in self.classifier.classes_] + [len('Flag')])
                val_length = np.maximum(len(str(np.max(cm)))+2, label_length)
                length_str = f':>{val_length}'
                row_format = f"{{:>{label_length}}}" + f"{{{length_str}}}" * (len(classes))
                message += '\n' +  '--------'
                if self.target_names is None:
                    message += '\n' +  f'Target {ind}'
                else:
                    message += '\n' +  f'Target: {self.target_names[ind]}'
                message += '\n' +  row_format.format("", ' Prediction', *[""]*(len(classes)-1))
                message += '\n' +  row_format.format("Flag", *classes)
                for metric, row in zip(classes, cm):
                    message += '\n' +  row_format.format(metric, *row)
            self.logger.info(message)
        else:
            return cms


    def calculate_all_scores(self, x, y):
        """Calculate all scores for all targets"""
        scores = {}
        for metric, metric_fun in self.scores_dict.items():
            scores[metric] = self.calc_score(x, y, metric_fun)
        return scores
    

    def print_performance(self, title, score_dict, num_decimals: int = 4):
        print_performance(self.target_names, title, score_dict, num_decimals)
