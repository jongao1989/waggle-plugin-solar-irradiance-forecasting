# Training

## General

```data_utility``` provides the needed code to load and preprocess data, and ```training_utility``` provides the code to train and optimize models. For the simplest explanation, see the ```interactive_explanation.ipynb``` notebook in the ```code``` folder for a simple, interactive explanation on how to train and optimize models. 

Models are trained using time series cross validation and early stopping. If you would like to change what hyperparameters you optimize, change the function ```training_utility.build_model()```. See the optuna documentation for more details.

## Provided models

All provided models were trained and optimized using the default code in ```training_utility.build_model()``` and the default values ```training_utility.OptimizationInfo```. All models were optimized over 20 trials, except for the non-seasonal, one hour model, which was optimized two separate studies of 20 trials each. See the optuna documentation for more details.