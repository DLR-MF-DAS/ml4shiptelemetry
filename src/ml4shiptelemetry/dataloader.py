import pandas as pd
import numpy as np
import logging

TIME_COLS = ['year', 'month', 'day', 'hour', 'minute']
TARGETS_REG = ['Temp [°C]', 'Cond [S/m]', 'Temp_int [°C]', 'Sal']
TARGETS_CLASS = ['Temp_Flag', 'Sal_Flag']
FLAG_GROUP = {'good': [1, 2], 'bad': [3, 4], 'ignore': [0, 5, 6, 7, 8, 9]}
FLAG_GROUP_LABEL = {'good': 1, 'bad': 0, 'ignore': -1}


def preprocess_dat_file(dat_file_path, n_neighbours=0):
    df_dat = pd.read_csv(dat_file_path, sep='\t', encoding='ISO-8859-1', dtype=object, dtype_backend='pyarrow')

    # Remove non-informative ROWS and COLUMNS containing strings# Drop no longer needed columns
    cols_to_drop = ['RSWS.RSSYS.SNameSWMC', 'RSWS.RSSMC.SMCName']
    df_dat = df_dat.drop(0, axis=0).drop(1, axis=0).drop(columns=cols_to_drop)
    
    # Convert all columns to numerics, setting all non-numbers to NaN.
    for column_name in df_dat.columns:
        df_dat[column_name] = pd.to_numeric(df_dat[column_name], errors='coerce')

    # Replace faulty values with NaN.
    df_dat.replace(-987654, np.nan, inplace=True)

    # Add additional month column to keep in the independent variables
    df_dat['monthofyear'] = df_dat['month']

    # Add datetime column to calculate time between readouts.
    df_dat['datetime'] = pd.to_datetime(df_dat[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df_dat['time_to_prev'] = (df_dat['datetime'] - df_dat['datetime'].shift(1)).dt.total_seconds().fillna(0)

    # Sort data chronologically
    df_dat = df_dat.sort_values(by=TIME_COLS + ['second'])

    # Drop no longer needed columns
    cols_to_drop = ['second', 'datetime']
    df_dat = df_dat.drop(columns=cols_to_drop)

    # Aggregate dat data to minute resolution.
    df_dat_grouped = df_dat.groupby(TIME_COLS).mean()
    df_dat_grouped['max_time_to_prev'] = df_dat.groupby(TIME_COLS)['time_to_prev'].max()
    df_dat_grouped['median_time_to_prev'] = df_dat.groupby(TIME_COLS)['time_to_prev'].median()

    # Include previous values
    if n_neighbours > 0:
        df_prev = df_dat_grouped.shift([ind for ind in range(1, n_neighbours+1)], axis=0)
        df_next = df_dat_grouped.shift([-ind for ind in range(1, n_neighbours+1)], axis=0)
        df_dat_grouped = pd.concat([df_dat_grouped, df_prev, df_next], axis=1)
    return df_dat_grouped


def drop_columns_no_payload(file):
    cols_to_drop = ['MC', 'Latitude', 'Longitude', 'Depth water [m]', 'Temp_Flag', 'Sal_Flag']
    df = pd.read_csv(file, sep='\t', index_col='Date/Time', dtype_backend='pyarrow')
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df


def cleanse_nan_strings(df):
    # Convert NaN strings to numpy.NaN
    for column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df


def preprocess_tab_file(tab_file_path):
    # cols_to_drop = ['MC', 'Latitude', 'Longitude', 'Depth water [m]', 'Temp_Flag', 'Sal_Flag', 'Sal_Salinometer']
    cols_index = ['Date/Time']
    cols_all = cols_index + TARGETS_REG + TARGETS_CLASS
    df_tab = pd.read_csv(tab_file_path, sep='\t', usecols=cols_all, dtype='object')


    # Replace Date/Time column with multiple columns with year, month, etc.
    date_times = pd.to_datetime(df_tab['Date/Time'], format='%Y-%m-%dT%H:%M:%S')

    # Sort by time to allow preprocessing
    df_tab = df_tab.sort_values(by='Date/Time')
    
    df_tab['year'] = date_times.dt.year
    df_tab['month'] = date_times.dt.month
    df_tab['day'] = date_times.dt.day
    df_tab['hour'] = date_times.dt.hour
    df_tab['minute'] = date_times.dt.minute

    # Drop original Date/Time column
    # df_tab = df_tab.drop(columns=['Date/Time'])

    # Set index to Date/Time to allow time-based interpolation
    df_tab = df_tab.set_index('Date/Time')

    # Convert all columns to numeric (Doesn't affect index)
    for column_name in df_tab.columns:
        # Convert NaN strings to numpy.NaN
        df_tab[column_name] = pd.to_numeric(df_tab[column_name], errors='coerce')

    # Regression: Replace NaN values with the average between its previous and its next well-defined value
    df_tab[TARGETS_REG] = df_tab[TARGETS_REG].interpolate()

    # Classification: 
    # Replace NaN values with value 9 as given in DAM Data Processing Report
    # Map flags to class labels
    for column_name in TARGETS_CLASS:
        df_tab[column_name] = df_tab[column_name].fillna(9)

    # Drop datetime index, no longer needed
    df_tab = df_tab.reset_index(drop=True)

    # Calculate mean values per minute for regression targets and mode flag for classification targets
    df_tab_grouped = df_tab.groupby(TIME_COLS)[TARGETS_REG].mean()
    df_tab_grouped[TARGETS_CLASS] = df_tab.groupby(TIME_COLS)[TARGETS_CLASS].agg(pd.Series.mode)

    # Group flags
    for t in TARGETS_CLASS:
        df_tab_grouped.loc[df_tab_grouped[t].isin(FLAG_GROUP['good']), t+'_group'] = FLAG_GROUP_LABEL['good']
        df_tab_grouped.loc[df_tab_grouped[t].isin(FLAG_GROUP['bad']), t+'_group'] = FLAG_GROUP_LABEL['bad']
        df_tab_grouped.loc[df_tab_grouped[t].isin(FLAG_GROUP['ignore']), t+'_group'] = FLAG_GROUP_LABEL['ignore']
        # Remove ignored flags
        df_tab_grouped = df_tab_grouped[df_tab_grouped[t+'_group'].isin([FLAG_GROUP_LABEL['good'], FLAG_GROUP_LABEL['bad']])]
        # Overwrite original target column
        df_tab_grouped[t] = df_tab_grouped[t+'_group']
        # Delete new target column
        df_tab_grouped = df_tab_grouped.drop(columns=[t+'_group'])
        df_tab_grouped[t] = df_tab_grouped[t].astype(int)
    
    # If there still are missing (NaN) values, drop these
    df_tab_grouped = df_tab_grouped.dropna()
    return df_tab_grouped


def align_dat_with_tab(dat, tab):    
    # Ensure all rows in dat have a corresponding row in tab and vice versa
    tab = tab[tab.index.isin(dat.index)]
    dat = dat[dat.index.isin(tab.index)]

    # Sort the dat to same order as tab. Throws KeyError in case tab has indices that do not exist in dat.
    dat = dat.loc[tab.index, :]

    return dat, tab


def split_reg_class(df):
    """Split data from tab files into classification and regression parts"""
    df_reg = df[TARGETS_REG].copy()
    df_class = df[TARGETS_CLASS].copy()
    return df_reg, df_class


#THIS WOULD BE THE IMPLEMENTATION IN CASE .DAT AND .TAB FILES SHARE THE SAME NAMES. E.G. FILE1.DAT, FILE1.TAB, ..., FILE-N.DAT, FILE-N.TAB
"""
def process_files(data_path):
    x = np.array([]).reshape(0, 0)  # Initializing empty array for x
    y = np.array([]).reshape(0, 0)  # Initializing empty array for y

    for file_name in os.listdir(data_path):
        if file_name.endswith('.dat'):
            dat_file = os.path.join(data_path, file_name)
            tab_file = os.path.join(data_path, file_name.replace('.dat', '.tab'))
            
            if os.path.exists(tab_file):
                preprocessed_df_dat = preprocess_dat_file(dat_file)
                preprocessed_df_tab = preprocess_tab_file(tab_file)

                preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

                if x.size == 0:
                    x = preprocessed_df_dat.values
                    y = preprocessed_df_tab.values
                else:
                    x = np.concatenate((x, preprocessed_df_dat.values), axis=0)
                    y = np.concatenate((y, preprocessed_df_tab.values), axis=0)

    return x, y
"""


def process_files(data_path, n_test_files=0, n_neighbours=0):
    "Load data and targets and put in training or test datasets."
    logger = logging.getLogger()

    # Files to load.
    # Add new files by adding tuples (dat-file, tab-file).
    data_files = [('/AI_RSWS_SYSTEM_WEATHER_MSM98/AI_RSWS_SYSTEM_WEATHER_MSM98.dat', '/MSM98_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM98_2/AI_RSWS_SYSTEM_WEATHER_MSM98_2.dat', '/MSM98_2_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM99_2/AI_RSWS_SYSTEM_WEATHER_MSM99_2.dat', '/MSM99_2_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM99/AI_RSWS_SYSTEM_WEATHER_MSM99.dat', '/MSM99_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM100/AI_RSWS_SYSTEM_WEATHER_MSM100.dat', '/MSM100_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM101/AI_RSWS_SYSTEM_WEATHER_MSM101.dat', '/MSM101_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM102/AI_RSWS_SYSTEM_WEATHER_MSM102.dat', '/MSM102_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM103/AI_RSWS_SYSTEM_WEATHER_MSM103.dat', '/MSM103_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSM104/AI_RSWS_SYSTEM_WEATHER_MSM104.dat', '/MSM104_surf_oce.tab'),
                  ('/AI_RSWS_SYSTEM_WEATHER_MSMX14/AI_RSWS_SYSTEM_WEATHER_MSMX14.dat', '/MSM-X14_surf_oce.tab')
                  ]

    dat_files = tuple()
    tab_files_reg = tuple()
    tab_files_class = tuple()

    # Number of files to use for training
    n_train_files = len(data_files) - n_test_files

    # Load data files
    for idata, data_file in enumerate(data_files):
        if idata < n_train_files:
            train_test_set = 'training'
        else:
            train_test_set = 'test'
        logger.info(f"Processing dat file {data_file[0].split('/')[-1]} and tab file {data_file[1].split('/')[-1]}. Added to {train_test_set} set.")
        dat_file = data_path + data_file[0]
        tab_file = data_path + data_file[1]
        
        preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
        preprocessed_df_tab = preprocess_tab_file(tab_file)
        
        preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

        preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
        
        dat_files += (preprocessed_df_dat.to_numpy(),)
        tab_files_reg += (preprocessed_df_tab_reg.to_numpy(),)
        tab_files_class += (preprocessed_df_tab_class.to_numpy(),)
    
    feature_names = preprocessed_df_dat.columns.to_lilst()

    # Create training dataset
    x_train = np.concatenate(dat_files[0:n_train_files], axis = 0)
    y_train_reg = np.concatenate(tab_files_reg[0:n_train_files], axis = 0)
    y_train_class = np.concatenate(tab_files_class[0:n_train_files], axis = 0)

    # Delete rows corresponding to missing values in the training set
    nan_indices = np.any(np.isnan(y_train_reg), axis=1) | np.any(np.isnan(y_train_class), axis=1)
    x_train = x_train[~nan_indices, :]
    y_train_reg = y_train_reg[~nan_indices, :]
    y_train_class = y_train_class[~nan_indices, :]

    # Create test dataset
    if n_test_files > 0:
        x_test = np.concatenate(dat_files[n_train_files:], axis = 0)
        y_test_reg = np.concatenate(tab_files_reg[n_train_files:], axis = 0)
        y_test_class = np.concatenate(tab_files_class[n_train_files:], axis = 0)

        # Delete rows corresponding to missing values in the test set
        nan_indices = np.any(np.isnan(y_test_reg), axis=1) | np.any(np.isnan(y_test_class), axis=1)
        x_test = x_test[~nan_indices, :]
        y_test_reg = y_test_reg[~nan_indices, :]
        y_test_class = y_test_class[~nan_indices, :]
    else:
        x_test = None
        y_test_reg = None
        y_test_class = None

    return {'x': x_train, 
            'y_reg': y_train_reg, 
            'y_class': y_train_class, 
            'x_test': x_test, 
            'y_test_reg': y_test_reg, 
            'y_test_class': y_test_class, 
            'targets_reg': TARGETS_REG, 
            'targets_class': TARGETS_CLASS,
            'feature_names': feature_names}


# Uncomment and update the code below in case different data files needs different processing.
"""
#THIS IMPLEMENTATION IS AWARE OF FILE NAMES
def process_files(data_path, n_neighbours=0):

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM98/AI_RSWS_SYSTEM_WEATHER_MSM98.dat'
    tab_file = data_path + '/MSM98_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = preprocessed_df_dat.to_numpy()
    y_reg = preprocessed_df_tab_reg.to_numpy()
    y_class = preprocessed_df_tab_class.to_numpy()

    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM98_2/AI_RSWS_SYSTEM_WEATHER_MSM98_2.dat'
    tab_file = data_path + '/MSM98_2_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM99_2/AI_RSWS_SYSTEM_WEATHER_MSM99_2.dat'
    tab_file = data_path + '/MSM99_2_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM99/AI_RSWS_SYSTEM_WEATHER_MSM99.dat'
    tab_file = data_path + '/MSM99_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM100/AI_RSWS_SYSTEM_WEATHER_MSM100.dat'
    tab_file = data_path + '/MSM100_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM101/AI_RSWS_SYSTEM_WEATHER_MSM101.dat'
    tab_file = data_path + '/MSM101_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM102/AI_RSWS_SYSTEM_WEATHER_MSM102.dat'
    tab_file = data_path + '/MSM102_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM103/AI_RSWS_SYSTEM_WEATHER_MSM103.dat'
    tab_file = data_path + '/MSM103_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM104/AI_RSWS_SYSTEM_WEATHER_MSM104.dat'
    tab_file = data_path + '/MSM104_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSMX14/AI_RSWS_SYSTEM_WEATHER_MSMX14.dat'
    tab_file = data_path + '/MSM-X14_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, n_neighbours=n_neighbours)
    preprocessed_df_tab = preprocess_tab_file(tab_file)
    
    preprocessed_df_dat, preprocessed_df_tab = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)

    preprocessed_df_tab_reg, preprocessed_df_tab_class = split_reg_class(preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y_reg = np.concatenate((y_reg, preprocessed_df_tab_reg.to_numpy()), axis = 0)
    y_class = np.concatenate((y_class, preprocessed_df_tab_class.to_numpy()), axis = 0)
    
    #############################
    
    # Delete rows corresponding to missing values in the training set
    nan_indices = np.any(np.isnan(y_reg), axis=1) | np.any(np.isnan(y_class), axis=1)
    x = x[~nan_indices, :]
    y_reg = y_reg[~nan_indices, :]
    y_class = y_class[~nan_indices, :]
    
    return x, y_reg, y_class, TARGETS_REG, TARGETS_CLASS
"""
