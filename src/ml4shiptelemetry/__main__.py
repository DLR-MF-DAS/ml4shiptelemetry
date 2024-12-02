
import click
import os
import sys
import numpy as np
import joblib
from .RFRegressorModel import RFRegressor, RFClassifier
from .dataloader import process_files
import logging
import datetime
import json

TOP_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
DEFAULT_LOG_DIR = os.path.abspath(os.path.join(TOP_DIR, 'logs'))
DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(TOP_DIR, 'output'))
DEFAULT_PROC_DATA_DIR = os.path.abspath(os.path.join(TOP_DIR, 'data'))
DEFAULT_CV_PARAMS = {
                "regression": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "max_features": ["sqrt"],
                    "criterion": ["friedman_mse"],
                    "max_depth": [None]
                },
                "classification": {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "max_features": ["sqrt"],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None]
                }
            }

time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%dT%H%M%S')
log_filename = f'ml4shiptelemetry_{time_str}.log'

# Set up logging function
def setup_logging(log_dir=DEFAULT_LOG_DIR):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_path = os.path.join(log_dir, log_filename)  # Log file path

    # Set up logging configuration
    logging.basicConfig(
        level=logging.DEBUG,  # Adjust log level (DEBUG, INFO, WARNING, etc.)
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path),  # Log to file
            logging.StreamHandler(sys.stdout)  # Log to console
        ]
    )

@click.command()
@click.option('--data-dir', required=True, help='Data directory contaning .tab files and folders with .dat files.')
@click.option('--cv', is_flag=True, default=False, type=bool, help='Cross-validate model performance on training data. Leave out to not crossvalidate.')
@click.option('--ts-cv', is_flag=True, default=False, type=bool, help='Use time series cross-validation instead of regular cross validation.')
@click.option('--cv-params', default=None, type=str, required=False, help='Path to json file containing cross-validation hyperparameters to exhaustively evaluate.')
@click.option('--n-test-files', default=0, type=int, required=False, help='Number of data files to use as test set. Test files are picked from the back of the list of files.')
@click.option('--n-neighbours', default=0, type=int, required=False, help='Number of neighbouring rows to add to each row in the training set. For example, a value 1 means adding the row before and after.')
@click.option('--preprocessed-dir', default=DEFAULT_PROC_DATA_DIR, type=str, required=False, help='Directory to store processed data for faster reprocessing next time.')
@click.option('--model-output-dir', default=DEFAULT_MODEL_DIR, type=str, required=False, help='Directory to store models as pickle files.')
@click.option('--log-dir', default=DEFAULT_LOG_DIR, type=str, required=False, help="Directory to save log file.")
@click.option('--verbose', default=1, type=int, required=False, help='Verbosity of scikit-learn RandomSearchCV.')
def main(data_dir, cv, ts_cv, cv_params, n_test_files, n_neighbours, 
         preprocessed_dir, model_output_dir, log_dir, verbose):
    
    setup_logging(log_dir)
    
    if ts_cv and not cv:
        cv = True

    if preprocessed_dir is not None:
        if not os.path.exists(preprocessed_dir):
            os.makedirs(preprocessed_dir)
        proc_file_path = os.path.join(preprocessed_dir, f'proc_files_n_test_files_{n_test_files}_n_neighbours_{n_neighbours}.npz')
        try:
            # Try loading preprocessed data.
            data = np.load(proc_file_path)
            message = f'Preprocessed data loaded from {proc_file_path}'
            logging.info(message)
        except:
            # If loading preprocessed data failes, load the raw data, preprocess and store.
            # Load data
            data = process_files(data_dir, n_test_files=n_test_files, n_neighbours=n_neighbours)
            np.savez(proc_file_path, **data)
            message = f'Preprocessed data saved to {proc_file_path}'
            logging.info(message)
    else:
        # Load data
        data = process_files(data_dir, n_test_files=n_test_files, n_neighbours=n_neighbours)

    x = data['x']
    y_reg = data['y_reg']
    y_class = data['y_class']
    x_test = data['x_test']
    y_test_reg = data['y_test_reg']
    y_test_class = data['y_test_class']
    targets_reg = data['targets_reg']
    targets_class = data['targets_class']

    # Create regressor
    rfregressor = RFRegressor(target_names=targets_reg, 
                              n_estimators=100, max_depth=None, 
                              criterion='friedman_mse', max_features='sqrt',
                              n_jobs=-1)

    # Create Classifier
    targets_class = targets_class
    n_classifiers = len(targets_class)
    rfclassifiers = {targets_class[ind]: {'index': ind, 
                                          'classifier': RFClassifier(target_names=[targets_class[ind]], 
                                                                     n_estimators=100, 
                                                                     max_depth=18,
                                                                     sampling_strategy='auto',
                                                                     replacement=True,
                                                                     bootstrap=False,
                                                                     n_jobs=-1)} for ind in range(n_classifiers)}

    if cv:
        if cv_params is not None:
            with open(cv_params) as json_data:
                cvp = json.load(json_data)
        else:
            cvp = DEFAULT_CV_PARAMS
        print(cvp)
        # Performs both crossvalidation and refits on the full training set.
        message = 'Regression cross validation'
        logging.info(message)
        # print(message)
        rfregressor.crossvalidate(x, y_reg, num_splits=5, time_series=ts_cv, 
                                  cv_params=cvp['regression'],
                                  shuffle=False, verbose=verbose, n_jobs=1)

        message = '\nClassification cross validation'
        logging.info(message)
        # print(message)
        for t, clf in rfclassifiers.items():
            message = f"Target: {t}"
            logging.info(message)
            # print(message)
            clf['classifier'].crossvalidate(x, y_class[:, clf['index']], num_splits=5, time_series=ts_cv, 
                                            cv_params=cvp['classification'],
                                            shuffle=False, verbose=verbose, n_jobs=1)

    # Train full models and evaluate on test set
    if x_test is not None:
        # Train
        if not cv:
            rfregressor.fit(x, y_reg)
            for t, clf in rfclassifiers.items():
                clf['classifier'].fit(x, y_class[:, clf['index']])

        # Evaluate and print performance
        message = '\nTest performance'
        logging.info(message)
        # print(message)
        scores_reg = rfregressor.calculate_all_scores(x_test, y_test_reg)
        rfregressor.print_performance('\nRegression test performance', scores_reg)

        scores_class = []
        for t, clf in rfclassifiers.items():
            scores_class = clf['classifier'].calculate_all_scores(x_test, y_test_class[:, clf['index']])
            clf['classifier'].print_performance(f'\nClassification test performance: {t}', scores_class)
            clf['classifier'].confusion_matrix(x_test, y_test_class[:, clf['index']], print_matrix=True)

    if model_output_dir is not None:
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
        file_append = f"_n_neighbours_{n_neighbours}"
        # Store regression model
        joblib.dump(rfregressor, os.path.join(model_output_dir, f"model_regression{file_append}.pkl"))
        # Store classification model(s)
        joblib.dump(rfclassifiers, os.path.join(model_output_dir, f"model_classification{file_append}.pkl"))


if __name__ == "__main__":
    main()