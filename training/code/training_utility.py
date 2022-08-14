import data_utility
import numpy as np
import optuna
import os
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
    
from keras.metrics import mean_squared_error
from keras.models import Sequential, clone_model
from keras.layers import LSTM, GRU, SimpleRNN, Dense

# Configure GPU growth. This fixed some bugs I got with GPU running out, but
# may not be needed for all.
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#==============================================================================
# CLASSES : DATA AND TRAINING ORGANIZATION
#==============================================================================
class TrainTestData():
    '''
    Holds training and testing data together.
    '''
    def __init__(self, X_train, X_test, y_train, y_test):
        '''
            X_train: Data fed into the model during training.
            X_test: Data fed into the model during testing.
            y_train: Data for comparing against predictions during training.
            y_test: Data for comparing against predictions during testing.
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

class ModelArchitecture():
    '''
    Class which holds various variables used throughout model
    '''
    def __init__(self, steps_in, steps_out, resample_rate_min,
        train_test_data=None):
        '''
            n_steps_in (int): The number of timestamps fed into the model
            n_steps_out (int): The number of timestamps predicted by the model
            resample_rate_min (int): Resample rate in minutes.
            train_test_data (TrainTestData): Training/testing data
            n_features (int): The number of features fed into the model. Set to
                2. Note that one feature is always predicted.   
        '''
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.resample_rate_min = resample_rate_min
        self.train_test_data = train_test_data
        self.n_features = 2

class OptimizationInfo():
    '''
    Class which holds various variables used for training/optimization
    '''
    def __init__(self, n_trials, n_splits=5, n_epochs=140, batch_size=-1,
        min_improvement=0, patience=5):
        '''
        n_trials: how many trials to optimize hte model over.
        n_splits: number of slpits for time series cross validaiton.
        n_epochs: number of epochs to train over.
        batch_size: batch size. If set to -1, will replace with the length of
            each day in the dataset.
        min_improvement: minimum improvement before early stopping. If < 0,
            does not implement early stopping
        patience (int): number of iterations to wait for improvement before
            early stopping
        '''
        self.n_splits = n_splits
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.min_improvement = min_improvement
        self.patience = patience
        self.n_trials = n_trials

#==============================================================================
# DATA LOADING
#==============================================================================
def get_data(dirname, arch, train_test_ratio, generate_new_data):
    '''
    Loads pre-processed data from file, scales, and splits into train/test.

    Args:
        dirname: Directory of data.
        steps_in: Length of input in steps.
        steps_out: Length of prediction in steps.
        resample_rate_min: Resample rate in minutes.
        train_test_ratio: Fraction of data to use for training.
        generate_new_data: If false, searches for pickled data files created by
            data_utility.py.
        
    Returns:
        train_test_data: TrainTestData holding training and testing data.
        scalers: Scalers for each feature.
        batch_size
    '''
    steps_in = arch.steps_in
    steps_out = arch.steps_out
    resample_rate_min = arch.resample_rate_min

    # load and scale data
    if generate_new_data:
        unscaled_X, unscaled_y = data_utility.main(steps_in, steps_out,
            resample_rate_min, dirname, write_to_file=True)
    else:
        unscaled_X, unscaled_y = open_data_from_file(dirname, steps_in,
            steps_out)
    X, y, scalers = scale_data(unscaled_X, unscaled_y)    
    
    # determine sizes for batches, train/test split
    hours_per_day = 14 # if using default day_time in training_utility.py
    datapoints_per_hour = 60/resample_rate_min
    datapoints_per_day = hours_per_day * datapoints_per_hour + 1
    n_data_points = X.shape[0] + (steps_in - 1)
    n_days = n_data_points / datapoints_per_day

    train_size = np.floor(n_days * train_test_ratio)/n_days
    batch_size = int(datapoints_per_day)

    # split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False,
        train_size=train_size)
    train_test_data = TrainTestData(X_train, X_test, y_train, y_test)
    arch.train_test_data = train_test_data
    return arch, scalers, batch_size

def open_data_from_file(dirname, steps_in, steps_out, n_features=2):
    '''
    Loads data from files in given directory.
    '''
    # Create empty arrays of the correct shape to fill with data.
    X = np.empty((0,steps_in,n_features))
    y = np.empty((0,steps_out))
    for file in os.listdir(dirname):
        # Iterate through each file and fill X, y with data.
        filename = os.path.join(dirname, file)
        if file.endswith(".X.pickle"):
            to_open = open(filename, 'rb')
            X = np.concatenate((X, pickle.load(to_open)))
            to_open.close()
        if file.endswith(".y.pickle"):
            to_open = open(filename, 'rb')
            y = np.concatenate((y, pickle.load(to_open)))
            to_open.close()
    return X, y

def scale_data(X, y):
    '''
    Scales each variable independently.

    Args:
        X: Data inputted to model.
        y: Data validated against.
    
    Returns:
        Scaled X and y.
        Scalers used for each feature.
    '''
    scalers = []
    for i in range(X.shape[-1]):
        scaler = MinMaxScaler()
        X[:,:,i] = scaler.fit_transform(
            X[:,:,i].reshape(-1,1)).reshape(X[:,:,i].shape)
        scalers.append(scaler)
    y = scalers[0].transform(y.reshape(-1,1)).reshape(y.shape)
    return X, y, scalers


#==============================================================================
# FUNCTIONS : DATA AND TRAINING ORGANIZATION
#==============================================================================
def train_and_evaluate_model(trial, arch, opt_info):
    '''
    Trains model and evaluates against test data.
    
    Args:
        trial (optuna trial): An attempt at finding the best performing model.
        arch (ModelArchitecture): Object containing frequently used vars
            (e.g. n_steps_in, n_features) and training/testing data

    Returns:
        model (keras model): Model, trained.
        loss (float): Mean squared error against test data, used to measure
            performance of model.
    '''
    # first, train the model (using time series cross validation)
    model, loss_history = train_model(trial, arch, opt_info)
    print("-----training complete. evaluating model...")

    # second, evaluate it to determine the loss
    loss, pred = eval_model(model, arch.train_test_data.X_test,
        arch.train_test_data.y_test, return_pred=True)
    loss = np.average(loss)
    print("-----model evaluated")

    # third, save the loss history in the trial
    trial.set_user_attr("loss_history", loss_history)
    return model, loss, pred

def train_model(trial, arch, opt_info):
    '''
    Primary method for training. Trains for given number of epochs with time
    series cross validation.
    
    Args:
        trial (optuna trial): Trial, which stores chosen hyperparamters and
            performance
        arch (Architecture): Object containing frequently used vars
            (e.g. n_steps_in, n_features) and training/testing data
        n_epochs (int): number of epochs to train for
        min_improvement (int): minimum improvement before early stopping.
            If < 0, does not implement early stopping
        patience (int): number of iterations to wait for improvement before
            early stopping

    Returns:
        model (keras model): Model, trained
        epoch_loss (float): Loss of each epoch
    '''
    print("-----begin training")
    best_model = build_model(trial, arch)
    epoch_loss = [np.inf]
    best_idx = 0
    for epoch in range(opt_info.n_epochs):
        # for each epoch:
        # 1) train the model using time series cross validation. for all epochs
        #    after the first, this is continuing the training of the previous
        #    epochs.
        # 2) decide whether or not to stop early depending on the improvement
        #    compared to previous epochs

        # train one epoch
        print("-----epoch #" + str(epoch+1))
        curr_model, mse = train_epoch(best_model, arch.train_test_data.X_train,
            arch.train_test_data.y_train, opt_info)
        epoch_loss.append(np.average(mse))

        # early stopping:
        if opt_info.min_improvement >= 0:
            curr_idx = len(epoch_loss) - 1
            print(f"loss={epoch_loss[-1]}, improvement, best_idx={best_idx},"
                  f"curr_idx={curr_idx}")

            if epoch_loss[best_idx] - epoch_loss[-1] >= opt_info.min_improvement:
                # if model improved >= min_improvement, update the best loss
                # and index
                best_idx, best_model = curr_idx, curr_model
            if curr_idx - best_idx > opt_info.patience:
                # if the model hasn't improved for enough epochs, stop early
                print("-----early stopping. best_idx=" + str(best_idx) +
                        ", curr_idx=" + str(curr_idx))
                epoch_loss = epoch_loss[:best_idx]
                break
    
    print(epoch_loss)
    return best_model, epoch_loss

def build_model(trial, arch):
    '''
    Builds a model for optimization with optuna. You can adjust which
    hyperparameters you want to tune here. Example tunes RNN type, number of
    layers, and number of neurons per layer.
    
    Args:
        trial (optuna trial): Trial, which stores chosen hyperparamters and
            performance.
        arch (Architecture): Object with info about model structure and
            training/testing data.
    
    Returns:
        model (keras model): Model, trained.
        loss (float): Mean squared error, used to measure performance of model.
    '''
    n_neurons = trial.suggest_categorical('n_neurons', [32, 64, 128]) # tune
    n_layers = trial.suggest_int('n_layers', 1, 4) # tune
    rnn_type = trial.suggest_categorical('cell_type', ["LSTM", "GRU",
                                                       "SimpleRNN"]) # tune

    if rnn_type == "LSTM":
        rnn = LSTM
    elif rnn_type == "GRU":
        rnn = GRU
    else:
        rnn = SimpleRNN

    # A lot of code, but essentially:
    # - If an RNN has another RNN after it, it must have return_sequences=True
    # - The first RNN should have input_shape = (steps_in, n_features)
    # - The last layer should be Dense(steps_out)
    model = Sequential()
    if n_layers == 1:
        model.add(rnn(n_neurons, activation="tanh",
            input_shape=(arch.steps_in, arch.n_features)))
    else:
        model.add(rnn(n_neurons, activation="tanh", return_sequences=True,
            input_shape=(arch.steps_in, arch.n_features)))
        for i in range(n_layers-2):
            model.add(rnn(n_neurons, activation="tanh",
                return_sequences=True))
        model.add(rnn(n_neurons))
    model.add(Dense(arch.steps_out))

    model.compile(optimizer="adam", loss="mse")
    return model

def train_epoch(model, X, y, opt_info):
    '''
    Trains one epoch of time series cross validation.
    
    Args:
        model (keras model): Current best model.
        X, y (np arrays): Training data, to be split for cross validation
        n_splits (int): number of splits for cross validation

    Returns:
        model (keras model): Model, trained
        loss (float): Average loss of all folds
    '''
    tscv = TimeSeriesSplit(opt_info.n_splits)
    fold_mse = []
    for train_index, val_index in tscv.split(X):
        # for each fold:
        # 1) make a copy of the model
        # 2) train and evaluate said model

        # get the fraction of the training data for training/eval the fold
        X_fold_train, X_fold_val = X[train_index], X[val_index]
        y_fold_train, y_fold_val = y[train_index], y[val_index]

        # make a new model with the same architecture and weights as the
        # current best model
        fold_model = clone_model(model)
        fold_model.set_weights(model.get_weights())

        # train and evaluate on the current fold
        trained_model, mse = train_fold(fold_model, X_fold_train, X_fold_val,
            y_fold_train, y_fold_val, opt_info.batch_size)
        fold_mse.append(mse)

    # return the trained model of the last fold (which has trained on all the
    # training data) and the average loss for each fold, which is treated as
    # the loss of the entire epoch
    return trained_model, np.average(fold_mse)

def train_fold(model, X_train, X_val, y_train, y_val, batch_size):
    '''
    Trains a model and evaluates it on the current fold of cross validation.
    
    Args:
        model (keras model): Model to be trained
        X_train, X_val, y_train, y_val (np arrays): Training/validation data
        batch_size (int): batch size
            
    Returns:
        model (keras model): Model, trained
        loss (float): mean squared error, used to measure performance of model
    '''
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, batch_size)
    loss = eval_model(model, X_val, y_val)
    return model, loss

def eval_model(model, X, y, return_pred=False):
    '''
    Evaluates a model.
    
    Args:
        model (keras model): Model to be trained
        X, y (np arrays): Evaluation data

    Returns:
        model (keras model): Model, trained
        loss (float): mean squared error, used to measure performance of model
    '''
    pred = model.predict(X)
    loss = mean_squared_error(pred, y)
    return loss, pred if return_pred else loss

#==============================================================================
# HYPERPARAMETER OPTIMIZATION
#==============================================================================
class Objective():
    '''
    Objective function to be optimized. Made as a class so we can pass data
    into the function
    
    Args:
        arch (ModelArchitecture): Object containing frequently used vars
            (e.g. n_steps_in, n_features) and training/testing data

    Returns:
        loss (float): mean squared error. Optuna will aim to optimize this
            metric.
    '''
    def __init__(self, arch, opt_info):
        self.arch = arch
        self.opt_info = opt_info

    TRIAL_MODELS = {}

    def __call__(self, trial):
        # This function is run when optimizing the model.
        model, loss, pred = train_and_evaluate_model(trial, self.arch,
            self.opt_info)
        self.TRIAL_MODELS[trial.number] = model

        return loss

def perform_study(arch, opt_info):
    '''
    Performs a study, which searches for the best model over the given
    number of trials. See Optuna documentation for more details.
    
    Args:
        n_trials (int): The number of trials for the study to run.
        arch (ModelArchitecture): Object containing frequently used vars
            (e.g. n_steps_in, n_features) and training/testing data

    Returns:
        study (optuna study): A study, which contains a number of trials with
            different hyperparameters
    '''
    objective = Objective(arch, opt_info)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, opt_info.n_trials)
    return study, objective.TRIAL_MODELS

#==============================================================================
# MASTER
#==============================================================================
def main(dirname, arch, opt_info, train_test_ratio=0.75,
    generate_new_data=False, export_folder=''):
    '''
    Master umbrella function. Loads data, performs hyperparameter optimization
    study, and then pickles both studies and scalers used to file.
    
    Args:
        dirname (str): Directory to save file to and load data from (if
            generate_new_data is True).
        steps_in (int): The number of steps fed into the model.
        steps_out (int): The number of steps predicted by the model.
        n_trials (int): Number of trials to find the best model.
        resample_rate_min (int): Time frequency to downsample data in minutes.
        train_test_ratio: What fraction of the data to use for training.
        generate_new_data: If false, searches for pickled data files created by
            data_utility.py.
        export_folder: Folder to save file to within dirname.

    Returns:
        None
    '''
    # get data and set variables based on that data
    arch, scalers, batch_size = get_data(dirname, arch, train_test_ratio,
        generate_new_data)
    if opt_info.batch_size == -1:
        opt_info.batch_size = batch_size
    print("-----data loaded")

    # train and optimize the model
    study, trial_models = perform_study(arch, opt_info)
    print("-----training and optimization done")

    # save results to file
    export_dirname = os.path.join(dirname, export_folder)
    save_best_model_to_file(study, trial_models, export_dirname)
    pickle_study_to_file(study, arch, export_dirname)
    pickle_scalers_to_file(scalers, export_dirname)

def save_best_model_to_file(study, trial_models, dirname):
    '''
    Saves the best performing model to a file using keras's built-in function.

    Args:
        study (optuna study): Containing all trials in a particular
            hyperparamter optimization search
        dirname (str): Directory to save file to
    
    Returns:
        None
    '''
    best_model = trial_models[study.best_trial.number]
    path = os.path.join(dirname, 'best_model')
    best_model.save(path)
    print("-----best model saved to " + path)

def pickle_study_to_file(study, arch, dirname):
    '''
    Pickles study to file, with file name of form "n_steps_in.n_steps_out.best_params". Pickles to the given directory
    For example "3in.2out.{n_layers=4, n_nodes=128}"
    
    Args:
        study (optuna study): Object containing all trials in a particular hyperparamter optimization search
        n_steps_in (int): The number of timestamps fed into the model
        n_steps_out (int): The number of timestamps predicted by the model
        dirname (str): Directory to save file to

    Returns:
        none
    '''
    filename = os.path.join(dirname,f'{arch.steps_in}in.{arch.steps_out}out.study.pickle')
    file = open(filename, 'wb')
    pickle.dump(study, file)
    file.close()

    print("-----study saved to " + filename)

def pickle_scalers_to_file(scalers, dirname):
    '''
    Pickles scalers to file, with file name "scalers" Pickles to the given directory
    
    Args:
        scalers (list of sklearn scalers): Scalers used to transform given dataset
        dirname (str): Directory to save file to

    Returns:
        none
    '''
    filename = os.path.join(dirname,"scalers.pickle")
    file = open(filename, 'wb')
    pickle.dump(scalers, file)
    file.close()

    print("-----scalers saved to " + filename)
    