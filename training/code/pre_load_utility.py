import numpy as np
import pickle
import xarray as xr

from data_utility import open_data, preprocess_data
from datetime import date
from sklearn.preprocessing import MinMaxScaler

#----------------------------------------------------------------------------------------------------#
# PRE-LOADING DATA                                                                                   #
#----------------------------------------------------------------------------------------------------#
def load_data(date_ranges):
    '''
    Opens x_array data given a list of date ranges. Uses self-written function from open_data.
    

            Parameters:
                    date_ranges (2d np array of datetime.date): Array of start (inclusive) and end
                        (exclusive) date ranges. Of shape (number of date ranges, 2)
                    

            Returns:
                    data (dataset): xr dataset of requested date ranges
    ''' 
    data = open_data(date_ranges)
    print("---------data loaded")
    return data



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

def data(n_steps_in, n_steps_out, date_ranges, resample="15Min", scale=False, write_to_file=True,
         prefix="../../!data/pre-loaded/"):
    '''
    Loads data and formats into time series. Scales, resamples, and writes to file as requested.
    

            Parameters:
                    n_steps_in (int): number of time steps to feed into model
                    n_steps_out (int): number of time steps for model to predict
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
    data = load_data(date_ranges)
    data = preprocess_data(data, resample)
    
    input_vars = ["downwelling_shortwave", "percent_opaque"]
    output_var = "downwelling_shortwave"
    X,y = format_time_series(data, input_vars, output_var, n_steps_in, n_steps_out)
    
    # scale the data
    if scale:
        scale_time_series_data(X,y)
    
    # write to file
    if write_to_file:
        pickle_to_file(X, y, date_ranges, n_steps_in, n_steps_out, resample, scale, prefix)