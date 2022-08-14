# Tensorflow Models and Studies

This folder holds tensorflow models (not tensorflow lite) and optuna studies, both of which have been saved via pickling. If the original tensorflow model is not provided, it can be extracted from the study using:

```study.best_trial.user_attrs['model']```

The loss can also be found by using:

```study.best_trial.value```

## Model-specific notes

One-hour, non-seasonal: loss (mse) = 0.004806

One-hour, seasonal: Each of the studies corresponds to a different season. Due to errors on my part, the identity of each study is uncertain, but can be determined by checking the loss:<br>
Spring: loss (mse) = 0.005548<br>
Summer: loss (mse) = 0.007214<br>
Fall: loss (mse) = 0.006212<br>
Winter: loss (mse) = 0.003121<br>

The loss for the one day and three hour models I unfortunately did not have time to check. Additionally, I did not have time to extract and convert the three-hour model. However, this process ir provided in the ```tf_to_tflite``` notebook.