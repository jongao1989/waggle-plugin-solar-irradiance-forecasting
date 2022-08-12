#!/usr/bin/env python

from training_utility import main

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

main(dirname='training/ex_training_data', n_trials=1, steps_in=16, steps_out=4,
    resample_rate_min=15, train_test_ratio=0.75, generate_new_data=False)


# Remember to configure the early stopping requirements and number of epochs.