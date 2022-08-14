# Tensorflow Models and Studies

This folder holds tensorflow models (not tensorflow lite) and optuna studies. If the original tensorflow model is not provided, it can be extracted from the study using:

```study.best_trial.user_attrs['model']```

The loss can also be found by using:

```study.best_trial.value```