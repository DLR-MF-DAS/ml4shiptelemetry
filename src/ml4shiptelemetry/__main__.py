import click
from .RFRegressorModel import RFRegressor
from .dataloader import process_files

@click.command()
@click.option('--data-dir', help='Data directory contaning .tab files and folders with .dat files', required=True)
@click.option('--ts_cv', is_flag=True, help='Use time series cross validation instead of regular cross validation', default=False, type=bool)
def main(data_dir, ts_cv):
    # Load data
    x, y, target_names = process_files(data_dir, sort_time=ts_cv)
    
    # Create regressor
    rfregressor = RFRegressor(100, 18, target_names=target_names)
    rfregressor.crossvalidate(x, y, num_splits = 5, time_series=ts_cv)

if __name__ == "__main__":
    main()