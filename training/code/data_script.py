#!/usr/bin/env python

from data_utility import main

main(steps_in=16, steps_out=4, resample_rate_min=15,
    dirname='training/ex_training_data', write_to_file=True)