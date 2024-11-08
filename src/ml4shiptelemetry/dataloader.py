import pandas as pd
import numpy as np

TIME_COLS = ['year', 'month', 'day', 'hour', 'minute']

def preprocess_dat_file(dat_file_path, sort_time=False):
    df_dat = pd.read_csv(dat_file_path, sep='\t', encoding='ISO-8859-1', dtype=object)
    
    # Remove non-informative ROWS and COLUMNS containing strings
    df_dat = df_dat.drop('RSWS.RSSYS.SNameSWMC', axis=1).drop('RSWS.RSSMC.SMCName', axis=1).drop(0, axis=0).drop(1, axis=0)
    
    # Convert NaN strings or general strings to numpy.NaN
    for column_name in df_dat.columns:
        df_dat[column_name] = pd.to_numeric(df_dat[column_name], errors='coerce')

    df_dat.replace(-987654, np.nan, inplace=True)

    # Sort data chronologically
    if sort_time:
        df_dat = df_dat.sort_values(by=TIME_COLS)

    df_dat_grouped = df_dat.groupby(TIME_COLS).mean()
    return df_dat_grouped

def drop_columns_no_payload(file):
    cols_to_drop = ['MC', 'Latitude', 'Longitude', 'Depth water [m]', 'Temp_Flag', 'Sal_Flag']
    df = pd.read_csv(file, sep='\t', index_col='Date/Time')
    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def cleanse_nan_strings(df):
    # Convert NaN strings to numpy.NaN
    for column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    return df

def preprocess_tab_file(tab_file_path, sort_time=False):
    cols_to_drop = ['MC', 'Latitude', 'Longitude', 'Depth water [m]', 'Temp_Flag', 'Sal_Flag', 'Sal_Salinometer']
    df_tab = pd.read_csv(tab_file_path, sep='\t')

    # Drop columns from df_tab if they exist.
    df_tab = df_tab.drop(columns=cols_to_drop, errors='ignore')

    # Replace Date/Time column with multiple columns with year, month, etc.
    date_times = pd.to_datetime(df_tab['Date/Time'], format='%Y-%m-%dT%H:%M:%S')
    
    df_tab['year'] = date_times.dt.year
    df_tab['month'] = date_times.dt.month
    df_tab['day'] = date_times.dt.day
    df_tab['hour'] = date_times.dt.hour
    df_tab['minute'] = date_times.dt.minute

    # Drop original Date/Time column
    df_tab = df_tab.drop(columns=['Date/Time'])

    #convert NaN strings to numpy.NaN
    for column_name in df_tab:
        df_tab[column_name] = pd.to_numeric(df_tab[column_name], errors='coerce')
    
    df_tab = df_tab.apply(pd.to_numeric, errors='coerce')
    # Replace NaN values with the average between its previous and its next well-defined value
    df_tab = df_tab.interpolate()
    
    # Sort data chronologically
    if sort_time:
        df_tab = df_tab.sort_values(by=TIME_COLS)

    df_tab_grouped = df_tab.groupby(TIME_COLS).mean()
    target_names = df_tab_grouped.columns.to_list()
    return df_tab_grouped, target_names

def align_dat_with_tab(dat, tab):
    dat_time_indexes = dat.index.tolist()
    tab_time_indexes = tab.index.tolist()
    
    for index in dat_time_indexes:
        if index not in tab_time_indexes:
            dat = dat.drop(index)
    return dat

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

#THIS IMPLEMENTATION IS AWARE OF FILE NAMES
def process_files(data_path, sort_time=False):

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM98/AI_RSWS_SYSTEM_WEATHER_MSM98.dat'
    tab_file = data_path + '/MSM98_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, target_names = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = preprocessed_df_dat.to_numpy()
    y = preprocessed_df_tab.to_numpy()

    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM98_2/AI_RSWS_SYSTEM_WEATHER_MSM98_2.dat'
    tab_file = data_path + '/MSM98_2_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM99_2/AI_RSWS_SYSTEM_WEATHER_MSM99_2.dat'
    tab_file = data_path + '/MSM99_2_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################
    
    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM99/AI_RSWS_SYSTEM_WEATHER_MSM99.dat'
    tab_file = data_path + '/MSM99_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM100/AI_RSWS_SYSTEM_WEATHER_MSM100.dat'
    tab_file = data_path + '/MSM100_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM101/AI_RSWS_SYSTEM_WEATHER_MSM101.dat'
    tab_file = data_path + '/MSM101_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM102/AI_RSWS_SYSTEM_WEATHER_MSM102.dat'
    tab_file = data_path + '/MSM102_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM103/AI_RSWS_SYSTEM_WEATHER_MSM103.dat'
    tab_file = data_path + '/MSM103_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSM104/AI_RSWS_SYSTEM_WEATHER_MSM104.dat'
    tab_file = data_path + '/MSM104_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    ##############################

    dat_file = data_path + '/AI_RSWS_SYSTEM_WEATHER_MSMX14/AI_RSWS_SYSTEM_WEATHER_MSMX14.dat'
    tab_file = data_path + '/MSM-X14_surf_oce.tab'
    
    preprocessed_df_dat = preprocess_dat_file(dat_file, sort_time=sort_time)
    preprocessed_df_tab, _ = preprocess_tab_file(tab_file, sort_time=sort_time)
    
    preprocessed_df_dat = align_dat_with_tab(preprocessed_df_dat, preprocessed_df_tab)
    
    x = np.concatenate((x, preprocessed_df_dat.to_numpy()), axis = 0)
    y = np.concatenate((y, preprocessed_df_tab.to_numpy()), axis = 0)
    
    #############################
    
    nan_indices = np.where(np.isnan(y))
    x = np.delete(x, nan_indices, axis=0)
    y = np.delete(y, nan_indices, axis=0)
    
    return x, y, target_names