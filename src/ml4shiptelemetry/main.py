import click
from RFRegressorModel import RFRegressor
from dataloader import process_files

@click.command()
@click.option('--data-dir', help='Data directory contaning .tab files and folders with .dat files', required=False)
def main(data_dir):
    rfregressor = RFRegressor(100, 18)
    #data_dir = '/localhome/zapp_an/Desktop/Vouchers/geomar_voucher/data/'
    x, y = process_files(data_dir)
    rfregressor.crossvalidate(x, y, num_splits = 5)

if __name__ == "__main__":
    main()