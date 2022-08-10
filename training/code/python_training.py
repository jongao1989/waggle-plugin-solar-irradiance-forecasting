#!/usr/bin/env python

from training_utility import optimize

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

n_trials=20
optimize("2022 summer argonne/jupyter/!data/pre-loaded/06all_data_12-hours-in_3-hours-out_15min/",
         n_steps_in=12*4, n_steps_out=3*4, n_trials=n_trials, resample_rate_min=15)


# Remember to configure the early stopping requirements and number of epochs. They may be set to epochs=1 for debugging
