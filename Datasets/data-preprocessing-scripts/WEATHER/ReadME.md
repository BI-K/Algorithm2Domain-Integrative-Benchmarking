# Algorithm2Domain - Datasets - WEATHER

This folder contains scripts that turn the data of the [Kaggle-Weather dataset](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather?select=weather_features.csv) into the file format required by the Algorithm2Domain Evalaution Framework. 

## How to apply the transformation scripts

1. Download the csv file from: 
2. Adapt the pre_processing.py-script.
   1. configure the DATA_PATH
   2. configure the ADATIME_SAVE_FOLDER
   3. configure the WINDOW_SIZE_HOURS and the PREDICTION_HOUR as well as the labels if you wish to


**Run the Code** - `python pre_processing.py` 


## Updates to the AdaTime script:

The following files are edited:

1. data_model_configs.py
   1. Added a new class with name OHIO where we declare the data and model configuration and parameters (took inspiration from the other existing classes and built one) - noticed that only a few configurations are mandatory and not all are required and if the mandatory are not provided the AdaTime throws an error.
   2. Declared 2 classes - ["Not Hypoglycemic", "Hypoglycemic"], declared 2 scenarios still have to understand what the scenario is 