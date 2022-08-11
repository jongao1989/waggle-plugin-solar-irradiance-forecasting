# Training

## Data Preprocessing


## Training & Optimization

### Notes
- This process is created to work with GPU-based computing (specifically, SWING @ Argonne).

See `training_utility.py`. The functions `main()` and `build_model()` are of key interest.

`main()` is the primary function: you'll give it the directory to the preprocessed data and several variables regarding how you want the training/optimization to be done. Specifically: