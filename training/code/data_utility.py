# 
#==============================================================================


import numpy as np
import pandas as pd
import os
import xarray as xr

from datetime import datetime, date, time, timedelta
from sklearn.preprocessing import MinMaxScaler

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
        resample_rate: The model's expected data resampling in minutes.
        input_vars: Names of the input variables from the data.
        output_vars: Name of the output variable from the data.
        '''
        self.dirname = dirname
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.resample_rate_min = resample_rate_min
        self.resample_code = f'{resample_rate_min}min'
        self.input_vars = ["downwelling_shortwave", "percent_opaque"]
        self.output_var = "downwelling_shortwave"
        self.n_features = 2

#----------------------------------------------------------------------------------------------------#
# OPEN DATA FROM FILE                                                                                #
#----------------------------------------------------------------------------------------------------#
# getting paths
def get_paths(basepath, path_heads, path_tails, date_ranges):
    '''
    Returns an np array of path names to open in multiple date range. Note that paths must
    follow the format Base-Head-YYYYMMDD-Tail (hyphens added for readibility).
    
            Parameters:
                    basepath (string): The base of the path, at the start
                    path_heads (array of strings): Array of path heads, after the basepath
                    path_tails (array of strings): Array of path tails, after the path head
                    date_ranges (2d np array of datetime.date): Array of start (inclusive) and end
                        (exclusive) date ranges. Of shape (number of date ranges, 2)
            Returns:
                    paths (ND nparray): np array of shape (number of path types, number of days)
    ''' 
    # SPECIAL CASE: if date ranges = none, then
    #         for sol irr, return the wildcard path
    #         for cloud cover, return the directory name
    # this difference is because of a bug in the cloud cover data with overlapping values, thus requiring manual opening of files
    if date_ranges == None:
        return np.array([basepath + "solar irradiance/*.nc", basepath + "tsi cloud cover/"])
    
    
    # get all of the paths for each individual date range; grouped as (date range, path type, date)
    paths = np.array([get_paths_indiv_date_range(basepath, path_heads, path_tails, date_range[0],
                                                 date_range[1]) for date_range in date_ranges])
    
    # reshape the array so grouped as (path type, date)
    reshaped_paths = np.array([paths[:,i,:].flatten() for i in range(paths.shape[1])])
    return reshaped_paths

def get_paths_indiv_date_range(basepath, path_heads, path_tails, date_start, date_end):
    '''
    Returns an np array of path names to open in a date range. Note that paths must
    follow the format Base-Head-YYYYMMDD-Tail (hyphens added for readibility).
    
    For example:
    basepath = "/data/"
    path_heads = ["sun_data.", "cloud_data."]
    path_tails = [".00000.csv", ".63300.csv"]
    date_start = June 1, 2015
    date_end = June 3, 2015
    -----
    paths = [["/data/sun_data.20150601.00000.csv", "/data/sun_data.20150602.00000.csv"],
             ["/data/cloud_data.20150601.63300.csv", "/data/cloud_data.20150602.63300.csv"]] 

            Parameters:
                    basepath (string): The base of the path, at the start
                    path_heads (array of strings): Array of path heads, after the basepath
                    path_tails (array of strings): Array of path tails, after the path head
                    date_start (datetime.date): Start date of desired paths, inclusive
                    date_end (datetime.date): End date of desired paths, exclusive
            Returns:
                    paths (ND nparray): np array of shape (number of path types, number of days)
    ''' 
    #----------------------------------------------------------------------------------------------------#
    # KNOWN ERRORS                                                                                       #
    #----------------------------------------------------------------------------------------------------#
    # Unfortunately, the TSI Cloud Cover data does not always end in the same 000000.cdf ending.
    # It seems that the winter months have variable numbers. For the current moment, we are only
    # looking at July, so this is not a problem, but this function will have to be rewritten
    # should we want to look at a selective subset of the data which includes data from (it looks
    # like) October 10 - February 9. Note that we could also just use /*.cdf to open all the data
    # if we aren't interested in a particular subset.
    #----------------------------------------------------------------------------------------------------#

    # end date must be after start date
    if date_end <= date_start:
        raise Exception("End date must be after start date")
    
    # must have equal number of path_heads and path_tails
    if len(path_heads) != len(path_tails):
        raise Exception("Must have equal number of path heads and tails")
    n_path_types = len(path_heads)
    
    # create np array to store paths in. note that we create it in format
    # [[paths of date1], [paths of date2], ...] and transpose it before returning
    # to be in format [[paths of format1], [paths of format 2], ...]
    paths = np.array([], dtype=str);
    
    n_days = (date_end-date_start).days
    curr_date = date_start
    for ndate in range(n_days):
        # create np array of new paths to add
        new_paths = np.array([basepath + path_heads[i] + curr_date.strftime("%Y%m%d") + path_tails[i]
                              for i in range(n_path_types)])
        paths = np.append(paths, new_paths)
        curr_date += timedelta(days=1)
        
    return paths.reshape((n_days,n_path_types)).T

# opening data
def open_data(date_ranges, basepath="../../!data/",
              path_heads=["solar irradiance/sgpradflux1longC1.c2.","tsi cloud cover/sgptsiskycoverC1.b1."],
              path_tails = [".060000.nc", ".000000.cdf"]):
    '''
    Opens and returns a dataset of solar irradiance and cloud coverage for the given date ranges

            Parameters:
                    date_ranges (2d np array of datetime.date): Array of start (inclusive) and end
                        (exclusive) date ranges. Of shape (number of date ranges, 2)
                    basepath (string): The base of the path, at the start
                    path_heads (array of strings): Array of path heads, after the basepath
                    path_tails (array of strings): Array of path tails, after the path head
            Returns:
                    merged_data (dataset): xr dataset of requested date ranges, resampled
    ''' 
    # first, get paths of files to open
    paths = get_paths(basepath, path_heads, path_tails, date_ranges)
    print("---------paths to open determined")
    
    # open sol irr and cloud data, then merge them together
    sol_irr_label = "downwelling_shortwave"
    sol_irr_data = open_sol_irr_data(paths[0], sol_irr_label)
    print("---------sol irr data opened")
    cloud_labels = ["percent_opaque"]
    cloud_data = open_cloud_data(paths[1], cloud_labels)
    print("---------cloud coverage data opened")
    
    merged_data = combine_datasets(sol_irr_data, cloud_data, cloud_labels)
    print("---------data merged")
    
    # resample if requested and return
    return merged_data

def clean_nan_values(dataset):
    '''
    Cleans dataset by removing nan values.
    
            Parameters:
                    dataset (dataset): xr dataset
            Returns:
                    cleaned_dataset (dataset): xr dataset, cleaned
    ''' 
    return dataset.dropna("time")

def open_sol_irr_data(paths, var):
    '''
    Returns an xarray dataarray of data from the given file paths. Only opens data for given var
    
            Parameters:
                    path_heads (1D np array of strings): Array of paths to open
                    var (string): Variable to look at
            Returns:
                    data (dataarray): Dataarray for given var
    ''' 
    return clean_nan_values(xr.open_mfdataset(paths)[var])

def open_cloud_data(paths, var_labels, suffix=".cdf", allowed_qc=[0], clean=True):
    '''
    Returns an xarray dataset of data from the given file paths. Opens data for var_labels and
    their associated quality checks. Performs quality check.
    
            Parameters:
                    paths (1D np array of strings): Array of paths to open. If len == 1, then it is assumed to be a directory of files
                        to open.
                    var_labels (array of string): Variables to look at
                    suffix (string): suffix of files to open
                    allowed_qc (array of int): Allowed numbers in quality check
                    clean (bool): whether or not to clean the data. see clean function for details
            Returns:
                    data (dataset): Dataset for given variables and their quality checks
    ''' 
    qc_labels = ["qc_" + x for x in var_labels]
    all_labels = var_labels + qc_labels
    
    # there is a bug when using xr.open_mfdataset on all the files (overlapping timestamps I think).
    # thus, we manually open each file and merge them together
    if paths is not np.ndarray:
        # case 1: open all files in the directory)
        datasets = [xr.open_dataset(os.path.join(paths, filename.decode("utf-8")))[all_labels]
                    for filename in os.listdir(paths) if filename.decode("utf-8").endswith(suffix)]
        data = xr.concat(datasets, dim="time")
        # trim overlapping values
        _, index = np.unique(data['time'], return_index=True)
        data = data.isel(time=index)
    else:
        # case 2: only open certain paths
        data = xr.open_mfdataset(paths)[all_labels]
    
    # clean the data (remove rows where given variables == -100)
    if clean:
        for label in var_labels:
            data = data.where(data[label] != -100, drop=True)
    quality_check_clouds(data, qc_labels, allowed_qc)        
    return clean_nan_values(data)

def quality_check_clouds(ds, qc_labels, allowed_qc):
    '''
    Throws an exception if quality check includes numbers beyond those in allowed_qc
    
            Parameters:
                    ds (xarray dataset): Dataset of cloud data
                    qc_labels (array of string): Quality check labels
                    allowed_qc (array of int): Values allowed in quality check
            Returns:
                    none
    ''' 
    unallowed_qc = [x for x in list(range(16)) if x not in allowed_qc]
    for qc in qc_labels:
        if np.any(np.in1d(unallowed_qc, ds[qc].values)):
            raise ValueError("Cloud coverage data quality check error in " + qc)

def combine_datasets(sol_irr_data, cloud_data, cloud_labels):
    '''
    Combines sol_irr_data with cloud_data using inner join (intersection of time values).
    Additonally, only types the cloud data (not the quality checks).
    
            Parameters:
                    sol_irr_data (xarray dataarray): Datarray of solar irradiance data
                    cloud_data (xarray dataset): Dataset of cloud coverage data
                    cloud_labels (array of string): Cloud data labels                
            Returns:
                    ds (xarray dataset): Dataset of merged solar irradiance and cloud coverage data
    ''' 
    return xr.merge([sol_irr_data, cloud_data[cloud_labels]], join="inner")


#----------------------------------------------------------------------------------------------------#
# PREPROCESSING DATA                                                                                 #
#----------------------------------------------------------------------------------------------------#
def preprocess_data(data, preprocess_instr):
    '''
    Preprocesses data. This includes removing faulty dates of data, trimming and padding,
    interpolating missing values, resampling, and converting to time series.
    
            Parameters:
                    data (xarray dataset): Dataset to be trimmed, padded, and resampled.
                    preprocess_instr
            Returns:
                    data (pd dataframe): Trimmed and padded dataframe.
    ''' 
    # convert data from xarray dataset to pandas dataframe
    data = data.to_dataframe()
    
    # shift the data to be in oklahoma time (so days are not cut in half)
    data = data.shift(-6, freq='H')
    
    # trim and pad the data to be of consistent length
    data = trim_and_pad_data(data)
    
    # resample data (by mean)
    resample_rule = data.index.floor(preprocess_instr.resample_code)
    data = data.groupby(resample_rule).mean()
    
    # format to time series and scale
    X, y = format_time_series(data, preprocess_instr)
    X, y = scale_time_series_data(X, y)

    print("---------data preprocessed")
    return X, y
    

def trim_and_pad_data(data):
    '''
    Removes dates with large amounts of missing data, pads each day with 0-valued data and then trims so each day
    have consistent length. Assumes data is in minutes. Note that it returns a pandas dataframe
    
            Parameters:
                    data (xarray dataset): Dataset to be trimmed and padded.
            Returns:
                    data (pd dataframe): Trimmed and padded dataframe.
    ''' 
    # add the date and clock time as individual columns to assist in trimming and padding
    data["date"] = data.index.date
    data["ctime"] = data.index.time
    # trim and pad each day and then remove the group label "date"
    data = data.groupby("date").apply(trim_and_pad_group).droplevel("date")
    # fill any nan values linearly
    data = data.interpolate(method='linear')
    
    return data

def trim_and_pad_group(group, day_time=[time(5,30,0),time(19,30,0)], essential_time=[time(8,20,0),time(16,40,0)]):
    '''
    Removes dates with large amounts of missing data, pads each day with 0-valued data and then trims so each day
    have consistent length. Assumes data is in minutes.
    
            Parameters:
                    group (pd dataframe): Dataframe to be trimmed and padded. Assumed to be of one day.
                    day_time (array of times): time range to pad/trim each day to.
                    essential_time (array of times): if enough data is missing from this time range, drop the date.               
            Returns:
                    group (pd dataframe): Trimmed and padded dataframe.
    ''' 
    
    # day_mask: pad this range with 0 values to fill for shorter days. cut off data outside of this range
    day_mask = (group["ctime"] > day_time[0]) & (group["ctime"] < day_time[1])
    # essential mask: if a lot of this data is missing, remove this date from the data
    essential_mask = (group["ctime"] > essential_time[0]) & (group["ctime"] < essential_time[1])
    
    # drop days without enough data
    if should_drop_group(group, essential_mask, essential_time):    
        return
    # otherwise, pad and trim data to be of consistent length
    else:
        # trim and pad the data to be within the day_time range
        group = group[day_mask]
        group = pad_group(group, day_time)
        return group

def should_drop_group(group, essential_mask, essential_time, requirement=0.8):
    '''
    Returns true if the given dataframe is missing enough data in the given timeframe to be dropped.
    Assumes dataset is in minutes.
    
            Parameters:
                    group (pd dataset): Dataframe to consider. Assumed to be of one day.
                    essential_mask (mask): mask to apply to data, crops data to just essential_time.
                    essential_time (array of times): time range to consider.          
                    requirement (float): the percent of data needed to not be dropped (between 0 and 1, inclusive)
            Returns:
                    bool (bool): Whether or not to drop the dataframe.
    '''    
    return group[essential_mask].shape[0] < get_difference_minutes(essential_time) * requirement

def pad_group(group, day_time):
    '''
    Pads the given dataframe with 0's in the "day time" region so each day has a consistent number of data points.
    Does this by determining the number of timestamps needed to pad at beginning and end, creating dataframes with
    the needed amounts, and then concating them to the original dataframe.
    
            Parameters:
                    group (pd dataframe): Dataframe to padded. Assumed to be of one day.
                    day_time (array of times): time range to pad each day to.          
            Returns:
                    group (pd dataframe): Padded dataframe.
    '''
    curr_date = group["date"][0]
    # create a list of datetimes which need to be added
    pre_rng = pd.date_range(datetime.combine(curr_date, day_time[0]),
                            datetime.combine(curr_date, group["ctime"][0]),
                            freq='1min', inclusive="left")
    post_rng = pd.date_range(datetime.combine(curr_date, group["ctime"][-1]),
                             datetime.combine(curr_date, day_time[-1]),
                             freq='1min', inclusive="right")
    # make datasets from the list of datetimes and concat them to the original data
    pre_df = pd.DataFrame({'downwelling_shortwave': [0] * len(pre_rng), 'percent_opaque': [0] * len(pre_rng)}, index=pre_rng)
    post_df = pd.DataFrame({'downwelling_shortwave': [0] * len(post_rng), 'percent_opaque': [0] * len(post_rng)}, index=post_rng)
    group = pd.concat([pre_df,group,post_df])    
    
    # create any missing indexes (e.g., randomly-missing hour). values currently = nan but will be fixed later
    full_indexes = pd.date_range(datetime.combine(curr_date, day_time[0]), datetime.combine(curr_date, day_time[1]),
                                 freq='1min')
    group = group.reindex(full_indexes)
        
    return group

def format_time_series(data, preprocess_instr):
    '''
    Formats the given data into a time series. Note this only permits one output variable.
    

            Parameters:
                    data (pandas): pandas dataframe of requested date ranges
                    preprocess_instr
                    

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

def scale_time_series_data(X,y):
    '''
    Scales time series data by individual variable.
    

            Parameters:
                    X (np array): data to input to model of requested variables,
                                  of shape (n_samples, n_steps_in, n_features_in)
                    y (np array): data for model to predict, of requested variable,
                                  of shape (n_samples, n_steps_out)
                    

            Returns:
                    None
    ''' 
    for i in range(X.shape[-1]):
        scaler = MinMaxScaler()
        X_1D = X[:,:,i]
        y_1D = y[:,i]
        X_1D = scaler.fit_transform(X_1D.reshape(-1,1)).reshape(X_1D.shape)
        y_1D = scaler.transform(y_1D.reshape(-1,1)).reshape(y_1D.shape)
        X[:,:,i] = X_1D
        y[:,i] = y_1D
    print("---------data scaled")
    return X,y

# helper functions
def get_difference_minutes(times):
    '''
    Gets the difference (not absolute) between times[1] and times[0]
    
            Parameters:
                    times (array of times): Times to find difference between.
            Returns:
                    difference (float): Difference in minutes between the times. May be negative.
    '''  
    return divmod((datetime.combine(date.min, times[1]) - datetime.combine(date.min, times[0])).total_seconds(), 60)[0]

#----------------------------------------------------------------------------------------------------#
# MAIN                                                                                               #
#----------------------------------------------------------------------------------------------------#
def main(steps_in, steps_out, resample_rate_min, date_ranges=None, scale=True, write_to_file=True,
         dirname="../ex_data/pre-loaded/"):
    '''
    Loads data and formats into time series. Scales, resamples, and writes to file as requested.
    
        Parameters:
            steps_in (int): number of time steps to feed into model
            steps_out (int): number of time steps for model to predict
            date_ranges (2d np array of datetime.date): Array of start (inclusive) and end
                (exclusive) date ranges. Of shape (number of date ranges, 2)
            resample (datetime format code as str): time frequency to downsample data. see
                https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes 
            scale (bool): whether to scale data or not
            write_to_file (bool): whether to pickle data to file or not
            prefix (str): prefix for filepaths
                    
        Returns:
            None
    ''' 

    preprocess_instr = PreprocessingInstructions(dirname, steps_in, steps_out, resample_rate_min, scale)

    data = load_data(date_ranges)
    X, y = preprocess_data(data, preprocess_instr)
    
    # write to file
    if write_to_file:
        pickle_to_file(X, y, date_ranges, steps_in, steps_out, resample, scale, dirname)

def pickle_to_file(X, y, date_ranges, n_steps_in, n_steps_out, resample, scale, prefix):
    '''
    Pickles data to file, with file name of form "YYYYMMDD-YYYYMMDD.n_steps_in.n_steps_out.resample_rate.scaled/unscaled.X/y"
    For example "20160701-20160801.3.2.15min.unscaled.X"
    

            Parameters:
                    date_ranges (2d np array of datetime.date): Array of start (inclusive) and end
                                (exclusive) date ranges. Of shape (number of date ranges, 2)
                    n_steps_in (int): number of time steps to feed into model
                    n_steps_out (int): number of time steps for model to predict
                    (datetime format code as str): time frequency to downsample data. see
                                https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes 
                    scale (bool): whether to scale data or not
                    prefix (str): prefix for filepaths
                    

            Returns:
                    None
    ''' 
    if date_ranges == None:
        date_ranges_str = "all_dates"
    else:
        date_ranges_str = '_'.join([date_range[0].strftime("%Y%m%d") + "-" + date_range[1].strftime("%Y%m%d")
                                 for date_range in date_ranges])
    steps_str = str(n_steps_in) + "." + str(n_steps_out)
    resample_str = "1min" if resample == None else resample
    scale_str = "scaled" if scale else "unscaled"
    filename = prefix + ".".join([date_ranges_str, steps_str, resample_str, scale_str])

    pickle.dump(X, open(filename + ".X", 'wb'))
    pickle.dump(y, open(filename + ".y", 'wb'))
    print("---------written to file, filename: " + filename + "." + " followed by X/y")