import numpy as np
from sklearn.preprocessing import MinMaxScaler

# global vars (elementary vars regarding data)
input_vars = ["downwelling_shortwave", "percent_opaque"]
output_var = "downwelling_shortwave"

# TODO: Rewrite all of this
# TODO: Should I just avoid np at all costs?

def format_time_series(data, n_steps_out):
    '''
    Formats the given data into a time series.
    
    Input data (X) is of the form: [n_time_series, n_steps_in, n_features]
    For example, if n_steps_in = 3, then [[1,2,3,4],[10,20,30,40]] would be converted into
    [[[1,10],[2,20],[3,30]],[[2,20],[3,30],[4,40]]]

    Output data (y) is of the form: [n_time_series, n_steps_out] (Assumes one feature out)
    For example, if n_steps_out = 3, then [1,2,3,4] would be converted into [[1,2,3],[2,3,4]]
    
    Args:
        data (pandas df): pandas dataframe of requested date ranges
        n_steps_out (int): number of time steps for model to predict
    Returns:
        X (np array): data to input to model of requested variables,
                        of shape (n_samples, n_steps_in, n_features_in)
        y (np array): data for model to predict, of requested variable,
                        of shape (n_samples, n_steps_out)
    '''
    # create empty X, y arrays with correct shape
    n_time_points = len(data[input_vars[0]])
    n_time_series = n_time_points-(n_steps_in + n_steps_out)+1
    X = np.empty((n_time_series, n_steps_in, n_features))
    y = np.empty((n_time_series, n_steps_out))

    # fill X, y. see above for explanation of what time series looks like
    for i in range(n_steps_out):
        y[:,i] = data[output_var][n_steps_in+i : n_time_points-n_steps_out+i+1]
    for i in range(n_time_series):
        X[i] = [[data[input_var][i + step] for input_var in input_vars] for step in range(n_steps_in)]
    return X,y

def scale_time_series_data(X, y, output_feature_idx):
    '''
    Scales time series data by individual feature.
    
    Args:
        X (np array): data to input to model of requested variables,
                        of shape (n_samples, n_steps_in, n_features_in)
        y (np array): data for model to predict, of requested variable,
                        of shape (n_samples, n_steps_out)
        output_feature_idx (int): index of feature to be predicted

    Returns:
        X, y (np arrays): X, y scaled
        scaler (sklearn minmaxscaler): scaler used to scale data. will later be used to inverse scale predictions
    ''' 
    scalers = []
    for i in range(X.shape[-1]):
        scaler = MinMaxScaler()
        X[:,:,i] = scaler.fit_transform(X[:,:,i].reshape(-1,1)).reshape(X[:,:,i].shape)
        y[:,i] = scaler.transform(y[:,i].reshape(-1,1)).reshape(y[:,i].shape)
        scalers.append(scalers)
    return X, y, scalers[output_feature_idx]

def preprocess_data(unprocessed_data, model_info):
    '''
    Converts unprocessed data into scaled, time series data.

    Args:
        unprocessed_data (???): solar irradiance and cloud cover data from the
            past 16 time steps
        args (argparse): program arguments

    Returns:
        scaled_time_series_data (np array): data formatted in time series and
            scaled by feature
    '''
    time_series_data = format_time_series(unprocessed_data,
                                                     model_info.steps_out)
    scaled_time_series_data = scale_time_series_data(
                              time_series_data)
    return scaled_time_series_data