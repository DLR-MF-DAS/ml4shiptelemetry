import click
import os
import numpy as np
import joblib
from .RFRegressorModel import RFRegressor, RFClassifier
from .dataloader import process_files

@click.command()
@click.option('--data-dir', required=True, help='Data directory contaning .tab files and folders with .dat files.')
@click.option('--cv', is_flag=True, default=False, type=bool, help='Crossvalidate model performance on training data. Leave out to not crossvalidate.')
@click.option('--ts_cv', is_flag=True, default=False, type=bool, help='Use time series cross validation instead of regular cross validation.')
@click.option('--n_test_files', required=False, default=0, type=int, help='Number of data files to use as test set. Test files are picked from the back of the list of files.')
@click.option('--n_neighbours', required=False, default=0, type=int, help='Number of neighbouring rows to add to each row in the training set. For example, a value 1 means adding the row before and after.')
@click.option('--classification_model', required=False, default='rf', type=str, help="Model to use for classification. Currently implemented: rf, balanced_rf.")
@click.option('--store_reuse', is_flag=True, default=False, type=bool, help='Store or load training and test data to file in the directory from which the python script is executed.')
@click.option('--model_output_path', default=None, type=str, required=False, help='Path to where models shall be stored pickle files at specified location. Leave out to not store models.')
def main(data_dir, cv, ts_cv, n_test_files, n_neighbours, classification_model, store_reuse, model_output_path):
    
    # If flag ts_cv is given, turn on crossvalidation even if the cv flag is not set.
    if ts_cv and not cv:
        cv = True

    if store_reuse:
        try:
            # Try loading preprocessed data.
            data = np.load(f'./proc_files_n_test_files_{n_test_files}_n_neighbours_{n_neighbours}.npz')
            x, y_reg, y_class, x_test, y_test_reg, y_test_class, targets_reg, targets_class = data['x'], data['y_reg'], data['y_class'], data['x_test'], data['y_test_reg'], data['y_test_class'], data['targets_reg'], data['targets_class']
        except:
            # If loading preprocessed data failes, load the raw data, preprocess and store.
            # Load data
            x, y_reg, y_class, x_test, y_test_reg, y_test_class, targets_reg, targets_class = process_files(data_dir,
                                                                                                            n_test_files=n_test_files,
                                                                                                            n_neighbours=n_neighbours)
            np.savez(f'./proc_files_n_test_files_{n_test_files}_n_neighbours_{n_neighbours}.npz', x=x, y_reg=y_reg, y_class=y_class, x_test=x_test, 
                    y_test_reg=y_test_reg, y_test_class=y_test_class, targets_reg=targets_reg, 
                    targets_class=targets_class)
    else:
        # Load data
        x, y_reg, y_class, x_test, y_test_reg, y_test_class, targets_reg, targets_class = process_files(data_dir,
                                                                                                        n_test_files=n_test_files,
                                                                                                        n_neighbours=n_neighbours)
    
    # Create regressor
    rfregressor = RFRegressor(100, 18, target_names=targets_reg)

    # Create Classifier
    rfclassifier = RFClassifier(100, 18, target_names=targets_class, model=classification_model)

    if cv:
        print('\nRegression cross validation')
        rfregressor.crossvalidate(x, y_reg, num_splits = 5, time_series=ts_cv, shuffle=False)

        print('\nClassification cross validation')
        rfclassifier.crossvalidate(x, y_class, num_splits = 5, time_series=ts_cv, shuffle=False)

    # Train full models and evaluate on test set
    if x_test is not None:
        # Train
        rfregressor.train(x, y_reg)
        rfclassifier.train(x, y_class)

        # Evaluate and print performance
        print('\nTest performance')
        scores_reg = rfregressor.calculate_all_scores(x_test, y_test_reg)
        rfregressor.print_performance('Regression test performance', scores_reg)

        scores_class = rfclassifier.calculate_all_scores(x_test, y_test_class)
        rfclassifier.print_performance('Classification test performance', scores_class)
        rfclassifier.confusion_matrix(x_test, y_test_class, print_matrix=True)

    if model_output_path is not None:
        # Store regression model
        joblib.dump(rfregressor, os.path.join(model_output_path, "model_regression.pkl"))
        # Store classification model
        joblib.dump(rfclassifier, os.path.join(model_output_path, "model_classification.pkl"))


if __name__ == "__main__":
    main()