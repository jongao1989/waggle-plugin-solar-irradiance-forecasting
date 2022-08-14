#!/usr/bin/env python

import training_utility

arch = training_utility.ModelArchitecture(steps_in=4, steps_out=1, resample_rate_min=30)
opt_info = training_utility.OptimizationInfo(n_trials=1, n_splits=5, n_epochs=1, min_improvement=0, patience=5)

training_utility.main('./training/ex_training_data', arch, opt_info, generate_new_data=True, export_folder='results')