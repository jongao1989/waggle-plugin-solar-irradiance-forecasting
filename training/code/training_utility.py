import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pickle
import tensorflow as tf

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    

    
from keras.metrics import mean_squared_error
from keras.models import Sequential, load_model, clone_model
from keras.layers import LSTM, GRU, SimpleRNN, Dense

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
    def __init__(self, steps_in, steps_out, n_features, train_test_data):
        '''
            n_steps_in (int): The number of timestamps fed into the model
            n_steps_out (int): The number of timestamps predicted by the model
            n_features (int): The number of features fed into the model. Note
                that one feature is always predicted.        
            train_test_data (TrainTestData): Training/testing data
        '''
        self.steps_in = steps_in
        self.steps_out = steps_out
        self.n_features = n_features
        self.train_test_data = train_test_data

class OptimizationInfo():
    '''
    Class which holds various variables used for training/optimization
    '''
    def __init__(self, n_splits=5, n_epochs=140, batch_size=57,
        min_improvement=0, patience=5, n_trials=20):
        '''
            
        '''
        self.n_splits = n_splits
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.min_improvement = min_improvement
        self.patience = patience
        self.n_trials = n_trials

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

    # second, evaluate it to determine the loss
    loss, pred = eval_model(model, arch.X_test, arch.y_test, return_pred=True)
    loss = np.average(loss)
    
    # third, save the loss history and model in the trial
    trial.set_user_attr("loss_history", loss_history)    
    trial.set_user_attr("model", model)  

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
            If == -1, does not implement early stopping
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
        curr_model, mse = train_epoch(best_model, arch.X_train, arch.y_train)
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
            input_shape=(arch.steps_in,arch.n_features)))
    else:
        model.add(rnn(n_neurons, activation="tanh", return_sequences=True,
            input_shape=(arch.steps_in,arch.n_features)))
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

    def __call__(self, trial):
        # This function is run when optimizing the model.
        model, loss, pred = train_and_evaluate_model(trial, self.arch,
            self.opt_info)
        return loss

def perform_study(n_trials, arch, opt_info):
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
    return study

def pickle_study_to_file(study, n_steps_in, n_steps_out, dirname):
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
    filename = f"{dirname}{n_steps_in}in.{n_steps_out}out.study"
    file = open(filename, 'wb')
    pickle.dump(study, file)
    file.close()

def pickle_scalers_to_file(scalers, dirname):
    '''
    Pickles scalers to file, with file name "scalers" Pickles to the given directory
    
            Parameters:
                    scalers (list of sklearn scalers): Scalers used to transform given dataset
                    dirname (str): Directory to save file to
            Returns:
                    none
    '''
    filename = f"{dirname}scalers"
    file = open(filename, 'wb')
    pickle.dump(scalers, file)
    file.close()
    
#==============================================================================
# MASTER
#==============================================================================
def optimize(dirname, n_steps_in, n_steps_out, resample_rate_min=15,
    n_features=2):
    '''
    Master umbrella function. Loads data, performs hyperparameter optimization
    study, and then pickles both studies and scalers used to file.
    
    Args:
        dirname (str): Directory to save file to
        n_steps_in (int): The number of timestamps fed into the model
        n_steps_out (int): The number of timestamps predicted by the model

    Returns:
        none
    '''
    # load data and store frequently-used variables and data in object
    X_train, X_test, y_train, y_test, scalers = get_data(dirname, n_steps_in, n_steps_out, resample_rate_min=resample_rate_min)
    train_test_data = TrainTestData(X_train, X_test, y_train, y_test)
    arch = ModelArchitecture(n_steps_in, n_steps_out, n_features, train_test_data)
    opt_info = OptimizationInfo()

    study = perform_study(n_trials, arch, opt_info)
    pickle_study_to_file(study, n_steps_in, n_steps_out, dirname)
    pickle_scalers_to_file(scalers, dirname)

def load_data_from_file(dirname, n_steps_in, n_steps_out, n_features=2):
    # load the data by iterating through all files
    X = np.empty((0,n_steps_in,n_features))
    y = np.empty((0,n_steps_out))
    for file in os.listdir(dirname):
        filename = dirname + file
        print(filename)
        if file.endswith(".X"):
            to_open = open(filename, 'rb')
            X = np.concatenate((X, pickle.load(to_open)))
            to_open.close()
        if file.endswith(".y"):
            to_open = open(filename, 'rb')
            y = np.concatenate((y, pickle.load(to_open)))
            to_open.close()
    return X, y

def scale_data(X, y):
    scalers = []
    for i in range(X.shape[-1]):
        scaler = MinMaxScaler()
        X[:,:,i] = scaler.fit_transform(X[:,:,i].reshape(-1,1)).reshape(X[:,:,i].shape)
        scalers.append(scaler)
    y = scalers[0].transform(y.reshape(-1,1)).reshape(y.shape)
    return X, y, scalers

def get_data(dirname, n_steps_in, n_steps_out, train_size_percent=0.75, resample_rate_min=15):
    # determine sizes for batches, train/test split
    hours_per_day = 14
    datapoints_per_hour = int(60/resample_rate_min)
    datapoints_per_day = hours_per_day * datapoints_per_hour + 1
    n_days = 1503 # manually defined, taken from looking at data
    train_size = np.floor(n_days * train_size_percent)/n_days

    # load, scale, and split data
    X, y = load_data_from_file(dirname, n_steps_in, n_steps_out)
    X, y, scalers = scale_data(X, y)    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, train_size=train_size)
    return X_train, X_test, y_train, y_test, scalers