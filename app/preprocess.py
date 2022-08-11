# Utility toolkit for preprocessing. This includes:
# 1) Scaling data
# 2) Formatting data into a time series
#
# Note that in training, data is first formatted into time series to avoid
# contaminating training data with test data. Because we are not training, we
# can scale first.
#==============================================================================


#==============================================================================
# PRELIMINARY: imports, global vars, etc.
#==============================================================================
import app
import numpy as np # TODO: Should we avoid np at all costs?
from sklearn.preprocessing import MinMaxScaler

#==============================================================================
# FUNCTIONS : PREPROCESSING
#==============================================================================
def scale_data(unprocessed_data):
    '''
    Scales data by individual feature.

    Args:
        unprocessed_data: Solar irradiance and cloud coverage data as a
            dictionary.

    Returns:
        scaled_data: Scaled solar irradiance and cloud coverage data as a
            dictionary.
        sol_irr_scaler (sklearn minmaxscaler): Scaler for solar irradiance.
    ''' 
    # scale each feature and store data and scaler in dictionaries
    scalers = {}
    scaled_data = {}
    for feat_name, feat_data in unprocessed_data.items():
        scaler = MinMaxScaler()
        scaled_feat_data = scaler.fit_transform(
            feat_data.reshape(-1,1)).reshape(feat_data.shape)
        scalers[feat_name] = scaler
        scaled_data[feat_name] = scaled_feat_data
    return scaled_data, scalers[app.TOPIC_HISTORICAL_SOL_IRR]

def format_time_series(data, steps_in):
    '''
    Formats the given data into a time series of form [steps_in, n_features_in]
    
    Args:
        data: Dictionary of scaled data.
        steps_in (int): The length of input in steps.

    Returns:
        Data formatted in a time series. Solar irradiance will be at index 0 in
        the second dimension.
    '''
    time_series = []
    for i in range(steps_in):
        time_series.append([data[app.TOPIC_HISTORICAL_SOL_IRR][i],
                            data[app.TOPIC_HISTORICAL_CLOUD_COVERAGE][i]])
    return time_series

def preprocess_data(unprocessed_data, model_info):
    '''
    Converts unprocessed data into scaled, time series data.

    Args:
        unprocessed_data: Solar irradiance and cloud coverage data as a
            dictionary of form: {TOPIC_HISTORICAL_SOL_IRR: sol_irr_data,
                                 TOPIC_HISTORICAL_CLOUD_COVERAGE: cloud_data}.
        model_info: Object with info about model structure. See ModelInfo.

    Returns:
        Preprocessed data (a scaled time series).
        Scalar used to scale solar irradiance.
    '''
    scaled_data = scale_data(unprocessed_data)
    scaled_time_series_data = format_time_series(scaled_data,
                                                 model_info.steps_in)
    return scaled_time_series_data