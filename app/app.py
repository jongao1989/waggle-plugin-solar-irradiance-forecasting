# Uses historical solar irradiance and cloud coverage to forecast solar
# irradiance.
#
# Plug-in works as follows:
# 1) Collects data for four hours
# 2) Predicts the next hour of data using the data collected in 1)
# 3) Publishes predictions
# 
# (Note that the model does NOT repeatedly make predictions; each time
# the plug-in is called, it collects data and makes a prediction once)
#==============================================================================

#==============================================================================
# IMPORTS
#==============================================================================
# imports for model
import numpy as np # can we optimize and get rid of this?
import os
import tensorflow.lite as tflite # TODO: Are things like os, time, argparse already installed?
from preprocess import preprocess_data
from time import sleep

# imports used for dummy data
import pickle # TODO: Should we install this in the docker container?

# imports for plug-in
import argparse
import global_constants as myconst
from waggle.plugin import Plugin

#==============================================================================
# CLASS : MODEL_INFO
#==============================================================================
class ModelInfo:
    '''
    Wrapper class for model directory and various variables regarding model
    architecture (i.e., # steps in/out, resample rate, etc.)
    '''
    def __init__(self, dirname, steps_in, steps_out, resample_rate,
                 is_seasonal_available, sol_irr_scaler=None):
        '''
        dirname: Directory of model. Assumes seasonal and non-seasonal models
            are in the same directory.
        steps_in: The length of input in steps.
        steps_out: The length of prediction in steps.
        resample_rate: The model's expected data resampling in minutes.
        is_seasonal_available: Whether seasonal models exist for the desired
            prediction length. All models must have non-seasonal model.
        sol_irr_scaler: Sklearn MinMax scaler for solar irradiance.
        n_features: Number of features fed into the model. Set to 2 (solar
            irradiance and cloud coverage).
        '''
        self.dirname = dirname
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.resample_rate = resample_rate
        self.is_seasonal_available = is_seasonal_available
        self.sol_irr_scaler = sol_irr_scaler
        self.n_features = 2
    
    # Static dict describing which month (by number) belongs in which season
    # { spr: mar-may, sum: jun-aug, fal: sep-nov, win: dec-feb,
    # non-seasonal: everything else (represented as -1) }
    SEASONS_DICT = dict(zip(np.concatenate([[12],range(1,12), [-1]]),
                            np.append(np.repeat(['winter','spring','summer',
                                                'fall'], 3),
                                      'non-seasonal')))

def get_model_info(predict_len_hours):
    '''
    Returns the relevant model for the desired number of hours to predict.
    Raises an error if a relevant model does not exist.

    Args:
        predict_len_hours: The length of prediction in hours
        
    Returns:
        ModelInfo for the relevant model.
    '''
    # dictionary with presets of available models
    MODELS_DICT = {1: ModelInfo('models/one_hour',
                                steps_in=16, steps_out=4, resample_rate=15,
                                is_seasonal_available=True),
                   24: ModelInfo('models/one_day',
                                steps_in=168, steps_out=28, resample_rate=30,
                                is_seasonal_available=False)}

    if predict_len_hours in MODELS_DICT.keys():
        return MODELS_DICT[predict_len_hours]
    else:
        raise ValueError('Model does not exist to predict the requested'
                         'number of hours.')

#==============================================================================
# FUNCTIONS : DATA COLLECTION
#==============================================================================
def get_data(plugin, model_info, input_path):
    '''
    Retrieve data from SAGE network. If dummy_data is provided, will instead
    retrieve dummy data from a file.
    
    Args:
        model_info: Object with info about model structure. See ModelInfo.
        input_path: Input path of data. If == None, collect live data from
            sensors.
    
    Returns:
        Preprocessed data.
    '''
    if input_path != None:
        # if using input data
        data = get_data_from_input_file(input_path)
    else:
        # if collecting live data from sensors
        data = collect_data_from_sensors(plugin, model_info)
    
    preprocessed_data, sol_irr_scaler = preprocess_data(data, model_info)
    model_info.sol_irr_scaler = sol_irr_scaler
    return preprocessed_data, model_info

def get_data_from_input_file(input_path):
    '''
    Opens a data structured as dictionary {TOPIC_HISTORICAL_SOL_IRR:
    sol_irr_data, TOPIC_HISTORICAL_CLOUD_COVERAGE: cloud_coverage_data}. See
    convert_to_pickled_dict.ipynb in the folder 'training' for details.
    '''
    with open(input_path, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict
  
def collect_data_from_sensors(plugin, model_info):
    '''
    Collect solar irradiance and cloud coverage data from sensors. Note that
    data is collected once each resample rate, so there is no actual
    resampling.

    Args:
        plugin: Waggle plugin.
        model_info: Object with info about model structure. See ModelInfo.
    
    Returns:
        Unprocessed data as a dict of form {data topic: data}
    '''
    plugin.subscribe(myconst.TOPIC_HISTORICAL_SOL_IRR,
        myconst.TOPIC_HISTORICAL_CLOUD_COVERAGE)

    sol_irr_data = []
    cloud_data = []

    # TODO: Repair this section. How to collect two data streams at once?

    for i in range(model_info.steps_in):
        msg = plugin.get()
        if msg.name == "env.solar.irradiance":
            sol_irr_data.append[msg.value]
        elif msg.name == "env.coverage.cloud":
            cloud_data.append[msg.value]
        sleep(model_info.resample_rate * 60)

    data = {myconst.TOPIC_HISTORICAL_SOL_IRR: sol_irr_data,
            myconst.TOPIC_HISTORICAL_CLOUD_COVERAGE: cloud_data}
    return data

#==============================================================================
# FUNCTIONS : FORECASTING
#==============================================================================
def get_model_path(model_info, month):
    '''
    Gets the path to the desired model. Assumes models are names in format:
    '{season}.tflite', e.g. 'fall.tflite', 'non-seasonal.tflite'. See
    SEASONS_DICT for the season names.

    Args:
        model_info: Object with info about model structure. See ModelInfo.
        month: Month of the first day of input data.
        
    Returns:
        Path to model.
    '''
    dirname = model_info.dirname
    suffix = '.tflite'
    season = ModelInfo.SEASONS_DICT[month]
    return os.path.join(dirname, f'{season}{suffix}')

def load_model(model_info, month):
    '''
    Create a tflite interpreter with the relevant model.

    Args:
        model_info: Object with info about model structure. See ModelInfo.
        month: Month of the first day of input data.
        
    Returns:
        The tflite interpreter created
    '''
    model_path = get_model_path(model_info, month)
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, input_data):
    '''
    Predicts the next values given input data. Number of time steps predicted
    dependent upon model.
    
    Args:
        interpreter: The tflite interpreter to predict the next values.
        input_data: Data to be used for prediction.
        
    Returns:
        The predicted output.
    '''
    output = interpreter.get_output_details()[0]
    input = interpreter.get_input_details()[0]
    input_data = np.expand_dims(input_data,axis=0)
    interpreter.set_tensor(input['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output['index'])

def convert_for_publish(predictions, scaler):
    '''
    Converts the predictions array for publication. This consists of:
    1) Inverse scaling the predictions
    2) Converting to bytes

    Args:
        predictions: Predictions as an array
        scaler: Scaler to inverse scale the predictions

    Returns:
        Inverse scaled predictions as bytes for publishing
    '''
    inv_scaled_predictions = scaler.inverse_transform(predictions)[0]
    print(inv_scaled_predictions)

    # TODO: Is an array an publishable type? Or does it need to be converted to
    #       bytes?
    return bytes(inv_scaled_predictions)

#==============================================================================
# FUNCTIONS : MAIN
#==============================================================================
def main(args):
    '''
    The "commander" function. Does all the actual work (i.e., getting and 
    preprocessing data, loading the model, predicting)

    Args:
        args: ArgumentParser of relevant arguments.

    Publish:
        Forecasted solar irradiance.
    '''
    with Plugin() as plugin:
        model_info = get_model_info(args.predict_len_hours)
        input_data, model_info = get_data(plugin, model_info, args.input_path)
        model = load_model(model_info, args.month)
        predictions = predict(model, input_data)
        publishable_predictions = convert_for_publish(predictions,
            model_info.sol_irr_scaler)

        plugin.publish(myconst.TOPIC_FORECASTED_SOL_IRR,
                       publishable_predictions)
    
if __name__ == '__main__':
    '''
    The chunk of code is run when this script is called. Sets up arguments for
    model to run and calls the functions which do the actual work.
    '''
    parser = argparse.ArgumentParser(description='''
                                     This program uses historical solar
                                     irradiance and cloud coverage data to
                                     predict future solar irradiance using a
                                     pre-trained model.''')

    parser.add_argument('-predict_len_hours', '-len', type=int,
                        help='How long of a prediction to make. An appropriate'
                             'model will be selected accordingly. See readme'
                             'for available lengths.')
    # TODO: If possible, determine month without user's direct input
    parser.add_argument('-month', '-m', type=int,
                        help='If using seasonal model: month of the year.'
                             'Model will assume non-seasonal if not specified',
                        default=-1)
    parser.add_argument('-input_path', '-i', type=str,
                        help='Path to input file. If not specified, the plugin'
                             'will take live data.',
                        default=None)
                        
    main(parser.parse_args())