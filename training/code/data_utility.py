# This file provides all the functions needed to open data from files using
# xarray and pickle them to file. It also completes most of the preprocessing
# (everything except for scaling).
#
# Assumes solar irradiance and cloud coverage folders are in the same folder
# and named as described by the global variables below.
#==============================================================================


#==============================================================================
# IMPORTS
#==============================================================================
import numpy as np
import pandas as pd
import os
import pickle
import xarray as xr

from datetime import datetime, date, time

SOL_IRR_FOLDER = 'sol_irr'
SOL_IRR_SUFFIX = '.nc'
CLOUD_COVERAGE_FOLDER = 'cloud_coverage'
CLOUD_COVERAGE_SUFFIX = '.cdf'

#==============================================================================
# CLASS : PREPROCESSING INSTRUCTIONS
#==============================================================================
class PreprocessingInstructions:
    '''
    Wrapper class for variables needed for preprocessing.
    '''
    def __init__(self, dirname, steps_in, steps_out, resample_rate_min):
        '''
        dirname: Directory of data. Assumes sol irr and cloud cover data are in
            the same directory.
        steps_in: The length of input in steps.
        steps_out: The length of prediction in steps.
        resample_rate_min: The model's expected data resampling in minutes.

        resample_code: The resample rate as a time code.
        input_vars: Names of the input variables from the data.
        output_vars: Name of the output variable from the data.
        sol_irr_input_idx: Index of solar irradiance in input_vars.
        n_features: How many features are fed into the model. Set to 2 (solar
            irradiance and cloud coverage)
        '''
        self.dirname = dirname
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.resample_rate_min = resample_rate_min

        self.resample_code = f'{resample_rate_min}min'
        self.input_vars = ["downwelling_shortwave", "percent_opaque"]
        self.output_var = "downwelling_shortwave"
        self.sol_irr_input_idx = 0
        self.n_features = 2

#==============================================================================
# FUNCTIONS : OPENING DATA
#==============================================================================
def open_data(preprocess_instr):
    '''
    Opens and returns a dataset of solar irradiance and cloud coverage data.

    Args:
        preprocess_instr: Wrapper class for variables needed for preprocessing.

    Returns:
        merged_data (dataset): xr dataset.
    ''' 
    # first, get wildcard paths of data
    paths = get_paths(preprocess_instr.dirname)
    print("---------paths to open determined")

    # open sol irr and cloud data, then merge them together
    sol_irr_label = preprocess_instr.input_vars[0]
    sol_irr_data = open_sol_irr_data(paths[SOL_IRR_FOLDER], sol_irr_label)
    print("---------sol irr data opened")

    cloud_label = preprocess_instr.input_vars[1]
    cloud_data = open_cloud_data(paths[CLOUD_COVERAGE_FOLDER], cloud_label)
    print("---------cloud coverage data opened")
    
    merged_data = combine_datasets(sol_irr_data, cloud_data, cloud_label)
    print("---------data merged")
    
    # resample if requested and return
    return merged_data

def get_paths(dirname):
    '''
    For solar irradiance, returns the wildcard path.
    
    For cloud coverage, returns the directory name. This is because of a bug in
    the cloud coverage data with overlapping values, thus requiring manual
    opening of files
    
    Args:
        dirname: Directory name of data.
                
    Returns:
        Dictionary of form {feature folder: path}
    ''' 

    return {SOL_IRR_FOLDER:
                os.path.join(dirname, SOL_IRR_FOLDER, f'*{SOL_IRR_SUFFIX}'),
            CLOUD_COVERAGE_FOLDER:
                os.path.join(dirname, CLOUD_COVERAGE_FOLDER)}

def open_sol_irr_data(wildcard_path, var):
    '''
    Returns an xarray dataarray of data from the given file paths. Only opens
    data for given var
    
    Args:
        wildcard_path: Array of wildcard paths to open.
        var: Name of variable to extract from data.
            
    Returns:
        Dataset of given variable for all files in the given wildcard path.
    ''' 
    return clean_nan_values(xr.open_mfdataset(wildcard_path)[var])

def open_cloud_data(cloud_dirname, var, allowed_qc=[0], clean=True):
    '''
    Returns an xarray dataset of data from the given directory. The TSI Cloud
    Coverage Data comes with a "quality check" which notes any errors. We thus
    open the data and perform a quality check.

    Note that the TSI Cloud Coverage Data has no data at night. In the times
    close to night, it has data = -100 (which can be removed).
    
    Parameters:
        cloud_dirname: Directory name of cloud coverage data.
        var: Name of variable to extract from data.
        allowed_qc (array of int): Allowed numbers in quality check.
        clean (bool): Whether or not to clean the data. see clean function for
            details

    Returns:
        Dataset of given variable for all files in the given wildcard path and
        its quality check.
    ''' 
    # Determine variable name of quality check.
    qc_label = "qc_" + var
    all_labels = [var, qc_label]
    
    # Manually open each file and merge together to avoid bug with
    # xr.open_mfdataset and overlapping values
    datasets = [xr.open_dataset(
        os.path.join(cloud_dirname, filename))[all_labels]
        for filename in os.listdir(cloud_dirname)
        if filename.endswith(CLOUD_COVERAGE_SUFFIX)]
    data = xr.concat(datasets, dim="time")

    # Trim overlapping values
    _, index = np.unique(data['time'], return_index=True)
    data = data.isel(time=index)
    
    # Clean the data (i.e., remove rows where given variables == -100)
    if clean:
        data = data.where(data[var] != -100, drop=True)
    quality_check_clouds(data, qc_label, allowed_qc)        
    return clean_nan_values(data)

def clean_nan_values(dataset):
    '''
    Cleans dataset by removing nan values.
    
    Args:
        dataset (dataset): xr dataset

    Returns:
        cleaned_dataset (dataset): xr dataset, cleaned
    ''' 
    return dataset.dropna("time")

def quality_check_clouds(ds, qc_label, allowed_qc):
    '''
    Throws an exception if quality check includes numbers beyond those in
    allowed_qc
    
    Parameters:
        ds (xarray dataset): Dataset of cloud data.
        qc_labels (array of string): Quality check labels.
        allowed_qc (array of int): Values allowed in quality check.

    Returns:
        None.
    ''' 
    unallowed_qc = [x for x in list(range(16)) if x not in allowed_qc]
    if np.any(np.in1d(unallowed_qc, ds[qc_label].values)):
        raise ValueError("Cloud coverage data quality check error in "
            + qc_label)

def combine_datasets(sol_irr_data, cloud_data, cloud_label):
    '''
    Combines sol_irr_data with cloud_data using inner join (intersection of
    time values). Note that cloud_data quality checks are removed.
    
    Args:
        sol_irr_data: Dataarray of solar irradiance data
        cloud_data: Dataset of cloud coverage data
        cloud_label: Variable name for cloud data. Used to remove quality
            checks.

    Returns:
        Xarray dataset of merged solar irradiance and cloud coverage data
    ''' 
    return xr.merge([sol_irr_data, cloud_data[cloud_label]], join="inner")

#==============================================================================
# FUNCTIONS : PREPROCESSING
#==============================================================================
def preprocess_data(data, preprocess_instr):
    '''
    Preprocesses data. This includes removing faulty dates of data, trimming
    and padding, interpolating missing values, resampling, and converting to
    time series. Note that scaling is not performed.

    Assumes dataset is in minutes.
    
    Args:
        data (xarray dataset): Dataset to be trimmed, padded, and resampled.
        preprocess_instr: Wrapper class for variables needed for preprocessing.

    Returns:
        Preprocessed (but not converted to timeseries nor scaled) data.
    ''' 
    # convert data from xarray dataset to pandas dataframe for ease of use.
    data = data.to_dataframe()
    
    # shift the data to be in oklahoma's time (so days are not cut in half)
    data = data.shift(-6, freq='H')
    
    # trim and pad the data so each day has consistent length
    data = trim_pad_interpolate_data(data)

    # resample data (by mean)
    resample_rule = data.index.floor(preprocess_instr.resample_code)
    data = data.groupby(resample_rule).mean()

    print("---------data preprocessed")
    return data

def trim_pad_interpolate_data(data):
    '''
    Removes dates with large amounts of missing data, pads each day with
    0-valued data and then trims so each day have consistent length. Assumes
    data is in minutes. Note that it returns a pandas dataframe
    
    Args:
        data (xarray dataset): Dataset to be trimmed and padded.
        
    Returns:
        data (pd dataframe): Trimmed and padded dataframe.
    ''' 
    # add the date and clock time as individual columns to assist in trimming and padding
    data["date"] = data.index.date
    data["ctime"] = data.index.time

    # trim and pad each day
    data = data.groupby("date").apply(trim_and_pad_day).droplevel("date")
    return data

def trim_and_pad_day(date_group):
    '''
    Given one day of data as a group, performs the following:
    1) Removes date if missing large amounts of data
    2) Pads with 0-valued data so each day will have consistent number of
       datapoints.
    3) Trims each day to a consistent time range. Each day thus has consistent
       length.
    
    Args:
        date_group: Dataframe to be trimmed and padded. Consists of one day.
                    
    Returns:
        group (pd dataframe): Trimmed and padded dataframe.
    ''' 
    # day_time: pad/trim each day to this time range
    # essential_time: if missing enough data from this time range, remove the
    #                 date from the dataset altogether.
    # note that these time ranges are slightly arbitrarily selected.
    day_time = [time(5,30,0),time(19,30,0)]
    essential_time = [time(8,20,0),time(16,40,0)]

    # create masks for each time range
    day_mask = (date_group["ctime"] > day_time[0]) & (
        date_group["ctime"] < day_time[1])
    essential_mask = (date_group["ctime"] > essential_time[0]) & (
        date_group["ctime"] < essential_time[1])
    
    # drop days without enough data
    if should_drop_day(date_group, essential_time, essential_mask):    
        return
    # otherwise, pad and trim data to be of consistent length
    else:
        # trim and pad the data to be within the day_time range
        date_group = date_group[day_mask]
        date_group = pad_and_interpolate_day(date_group, day_time)
        return date_group

def should_drop_day(date_group, essential_time, essential_mask,
    requirement=0.8):
    '''
    Returns true if the given dataframe is missing enough data in the given
    timeframe to be dropped. Assumes dataset is in minutes.
    
    Args:
        date_group: Dataframe to be trimmed and padded. Consists of one day.
        essential_time: If missing enough data from this time range, remove the
            date from the dataset altogether.
        essential_mask: Mask corresponding to essential_time. Crops data to
            just essential_time.
        requirement: the percent of data needed to not be dropped (between 0
            and 1, inclusive)

    Returns:
        Whether or not to drop the dataframe.
    '''    
    return date_group[essential_mask].shape[0] < get_difference_minutes(essential_time) * requirement

def pad_and_interpolate_day(date_group, day_time):
    '''
    Pads the given dataframe with 0's in the "day time" region so each day has
    a consistent number of data points. Does this by:
    1) determining the number of timestamps needed to pad at beginning and end
    2) creating dataframes with the needed amounts
    3) concating dataframes from 2) to the original dataframe.
    
    Args:
        date_group: Dataframe to be trimmed and padded. Consists of one day.
        day_time (array of times): Time range to pad each day to.

    Returns:
        date_group padded to fill day_time.
    '''
    curr_date = date_group["date"][0]

    # create a list of datetimes which need to be added
    pre_rng = pd.date_range(datetime.combine(curr_date, day_time[0]),
        datetime.combine(curr_date, date_group["ctime"][0]),
        freq='1min', inclusive="left")
    post_rng = pd.date_range(datetime.combine(curr_date, date_group["ctime"][-1]),
        datetime.combine(curr_date, day_time[-1]),
        freq='1min', inclusive="right")

    # make datasets from the list of datetimes and concat them to the original
    # dataset
    pre_df = pd.DataFrame({'downwelling_shortwave': [0]*len(pre_rng),
                           'percent_opaque': [0]*len(pre_rng)}, index=pre_rng)
    post_df = pd.DataFrame({'downwelling_shortwave': [0]*len(post_rng),
                            'percent_opaque': [0]*len(post_rng)}, index=post_rng)
    date_group = pd.concat([pre_df,date_group,post_df])    
    
    # create indexes for any missing values.
    full_indexes = pd.date_range(datetime.combine(curr_date, day_time[0]),
        datetime.combine(curr_date, day_time[1]), freq='1min')
    date_group = date_group.reindex(full_indexes)
        
    return interpolate_missing_data(date_group)

def interpolate_missing_data(data):
    '''
    Removes dates with large amounts of missing data, pads each day with
    0-valued data and then trims so each day have consistent length. Assumes
    data is in minutes. Note that it returns a pandas dataframe
    
    Args:
        data (xarray dataset): Dataset to be trimmed and padded.
        
    Returns:
        data (pd dataframe): Trimmed and padded dataframe.
    ''' 

    # fill any nan values linearly
    return data.interpolate(method='linear')

def format_time_series(data, preprocess_instr):
    '''
    Formats the given data into a time series.
    
    Parameters:
        data (pandas): pandas dataframe of requested date ranges
        preprocess_instr: Wrapper class for variables needed for preprocessing.
            
    Returns:
        X (np array): data to input to model of requested variables,
            of shape (n_samples, n_steps_in, n_features_in)
        y (np array): data for model to predict, of requested variable,
            of shape (n_samples, n_steps_out)
    ''' 
    steps_in = preprocess_instr.steps_in
    steps_out = preprocess_instr.steps_out

    # set up time series
    n_time_points = len(data[preprocess_instr.input_vars[0]])
    n_time_series = n_time_points-(steps_in + steps_out)+1

    X = np.empty((n_time_series, steps_in, preprocess_instr.n_features))
    y = np.empty((n_time_series, steps_out))

    for i in range(steps_out):
        y[:,i] = data[preprocess_instr.output_var][steps_in+i : n_time_points-steps_out+i+1]
    for i in range(n_time_series):
        X[i] = [[data[input_var][i + step] for input_var in preprocess_instr.input_vars] for step in range(steps_in)]
        print(str(i)+"/"+str(n_time_series), end="\r")
    print("---------time series set up")
    return X,y

def get_difference_minutes(times):
    '''
    Gets the difference (not absolute) between times[1] and times[0]
    
    Args:
        times (array of times): Times to find difference between.

    Returns:
        Difference in minutes between the times. May be negative.
    '''  
    return divmod((datetime.combine(date.min, times[1]) -
        datetime.combine(date.min, times[0])).total_seconds(), 60)[0]

# seasonal
def format_time_series_and_pickle_seasonal(data, preprocess_instr, seasons,
    write_to_file):
    '''
    '''
    time_series_data_by_season = {}
    for season_idx in range(len(seasons)):
        # for each season, get the data for that season
        season_data = data[data.index.month.isin(seasons[season_idx])]

        if not season_data.empty:
            # if data exists for that season, convert it to a time series and
            # pickle it if requested
            X, y = format_time_series_and_pickle_nonseasonal(season_data,
                preprocess_instr, write_to_file=False)
            time_series_data_by_season[season_idx] = {'X':X, 'y':y}

            if write_to_file:
                pickle_to_file(time_series_data_by_season[season_idx],
                    preprocess_instr, str(seasons[season_idx]))

    return time_series_data_by_season

def format_time_series_and_pickle_nonseasonal(data, preprocess_instr, write_to_file):
    '''
    '''
    X, y = format_time_series(data, preprocess_instr)

    if write_to_file:
        pickle_to_file({'X':X, 'y':y}, preprocess_instr)
    return X, y

#==============================================================================
# FUNCTIONS : MAIN
#==============================================================================
def main(steps_in, steps_out, resample_rate_min, dirname,
    seasons=None, write_to_file=True):
    '''
    Loads data from file, trims/pads, resamples, and formats into time series.
    Writes data to file as requested.
    
    Args:
        steps_in: Number of time steps to feed into model.
        steps_out: Number of time steps for model to predict.
        resample_rate_min: Time frequency to downsample data in minutes.
        dirname: Directory of data.
        seasons: List detailing which months are grouped together in a season.
            If provided, will divide data into seasons (for use with a seasonal
            model).
        write_to_file: whether or not to pickle data to file
                
    Returns:
        Unscaled time series data.
    ''' 

    preprocess_instr = PreprocessingInstructions(dirname, steps_in, steps_out,
        resample_rate_min)

    data = open_data(preprocess_instr)
    data = preprocess_data(data, preprocess_instr)


    # format to time series and write to file. do not scale since we haven't
    # split train/test. The process is different for seasonal/non-seasonal, so
    # there are two separate functions for htem.
    if seasons == None:
        return format_time_series_and_pickle_nonseasonal(data, preprocess_instr, write_to_file)
    else:
        return format_time_series_and_pickle_seasonal(data, preprocess_instr, seasons, write_to_file)

def pickle_to_file(things_to_pickle, preprocess_instr, season=''):
    '''
    Pickles each thing in given dictionary to file.
    
    Args:
        things_to_pickle: Dict of things to pickle, of form { name: thing }
        preprocess_instr: Wrapper class for variables needed for preprocessing.
        season: Name of the season, if using a seasonal model.
            
    Returns:
        None
    '''
    steps_in = preprocess_instr.steps_in
    steps_out = preprocess_instr.steps_out
    
    # steps, resample str
    data_info_str = '.'.join([f'{steps_in}in', f'{steps_out}out',
        preprocess_instr.resample_code])
    
    full_dirname = os.path.join(preprocess_instr.dirname, season)
    filename_base = os.path.join(full_dirname, data_info_str)
    for thing_name, thing in things_to_pickle.items():
        os.makedirs(full_dirname, exist_ok=True)
        thing_path = '.'.join([filename_base, season, thing_name, 'pickle'])
        pickle.dump(thing, open(thing_path, 'wb'))
        print("---------pickled to file, path: " + thing_path)