# Algorithm2Domain - Datasets - OHIOT1DM

This folder contains scripts that turn the data of the [OHIOT1D dataset](https://webpages.charlotte.edu/rbunescu/data/ohiot1dm/OhioT1DM-dataset.html) into the file format required by the Algorithm2Domain Evalaution Framework. 

## How to apply the transformation scripts

The following steps are taken:

- Used the original Python script pre_processing.py with minor changes and an update to save the data to .pt format.
  1. The utility functions remain the same form the original python script with a few changes:
     1. Added missing final chunk at the end of the for loop (mention with comment).
     2. Implemented chunk length check when performing smoothing.
     3. Implemented window-level z-score normalization as to perform the check to build class labels (1 = Hypo event, 0 = Normal).
     4. The threshold for Hypoglycemia event is 90.
  2. The process_file includes the following functions functionality (create_ground_truth and turn_into_samples) with the following changes:
     1. Conversion of data into .pt format.
     2. Also save the sample and labels into csv format for each train and test with respect to each patient.
     3. Please review if the process taken is correct for this or any changes are required.
     4. A sample .pt file is read from both the train and test file of the same patient and then is printed as well for checking and reference and to check the shape and unique labels and count of each unique label.

**Run the Code** - `python new_pre_processing.py` - Just data in the data directory is sufficient to run the script.

### File Structure

```bash
ADATIME_data/OHIO/
├── train_2018_559.pt
├── train_2018_563.pt
├── ... (other 2018 training files)
├── train_2020_540.pt
├── ... (other 2020 training files)
├── test_2018_559.pt
├── ... (other 2018 test files)
└── test_2020_540.pt
```

### Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `WINDOW_SIZE_MIN` | 60 | Input window length (minutes) |
| `PREDICTION_HORIZON_MIN` | 30 | Prediction horizon (minutes) |
| `HYPO_THRESHOLD` | 90 | Hypoglycemia threshold (mg/dL) |
| `max_missing_values_in_series` | 3 | Max allowed consecutive missing values |
| `smoothing_window` | 7 | Savitzky-Golay filter window size |


## Updates to the AdaTime script:

The following files are edited:

1. data_model_configs.py
   1. Added a new class with name OHIO where we declare the data and model configuration and parameters (took inspiration from the other existing classes and built one) - noticed that only a few configurations are mandatory and not all are required and if the mandatory are not provided the AdaTime throws an error.
   2. Declared 2 classes - ["Not Hypoglycemic", "Hypoglycemic"], declared 2 scenarios still have to understand what the scenario is 