# Solar Irradiance Forecasting

## Contents

This repository consists of a plug-in for the Waggle nodes (in ```app```), several pre-trained models ready for usage (in ```models```), and scripts to train new models (in ```training```).

### App

This folder includes a plug-in for use with the Waggle nodes to forecast future solar irradiance using historical solar irradiance and cloud coverage data. The plug-in can either load data from a file or take in streams of data from a Waggle sensor. You can also select how long of a forecast to predict (assuming there is a correlated tensorflowlite model). Information on how to use the plug-in as well as how it functions is fully documented in these files. Please note that the data needed to run these examples is not provided. See the Training section for information on what data is needed.

Due to timeline feasibilities, the plug-in was not able to be tested on an actual node. As a result, the sensor data collection methods may be buggy and the plug-in has not been tested in the docker environment. However, the plug-in is otherwise fully functional and ready for deployment.

### Models

This folder has pre-trained tensorflowlite models ready for use with ```app```, as well as the original, uncompressed tensorflow models and the optuna studies (which record the optimization process) for bookkeeping. Within ```models```, the folder ```tensorflow_models_and_studies``` has a readme document briefly explaining the most important functions for these studies. Further explanation can be found in the optuna documentation.

### Training

If you would like to train additional models, this folder provides the necessary scripts and explanation to do so. In ```training/code```, the notebook ```interactive_explanation``` has an interactive explanation on how to train additional models. This includes both non-seasonal and seasonal models, and can be configured to train models of various input and output lengths. Please note that the data needed for training (and thus for the examples) is not provided, though information on what data was used is in this the aforementioned notebook.

The models existing in this repository were trained time series cross validation and early stopping and optimized using optuna. See the readme document in the Training folder for further details.