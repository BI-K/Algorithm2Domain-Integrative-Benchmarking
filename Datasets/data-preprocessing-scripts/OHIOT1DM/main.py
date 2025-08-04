import os

import pre_processing



########################################## Config ####################################################################
# paths to training data
paths_training_data_2018 = ['../data/OhioT1DM_preprocessed/2018/train/559/', '../data/OhioT1DM_preprocessed/2018/train/563/', '../data/OhioT1DM_preprocessed/2018/train/570/', '../data/OhioT1DM_preprocessed/2018/train/575/', '../data/OhioT1DM_preprocessed/2018/train/588/', '../data/OhioT1DM_preprocessed/2018/train/591/']
paths_training_data_2020 = ['../data/OhioT1DM_preprocessed/2020/train/540/', '../data/OhioT1DM_preprocessed/2020/train/544/', '../data/OhioT1DM_preprocessed/2020/train/552/', '../data/OhioT1DM_preprocessed/2020/train/567/', '../data/OhioT1DM_preprocessed/2020/train/584/', '../data/OhioT1DM_preprocessed/2020/train/596/']

# path to test data
paths_test_data_2018 = ['../data/OhioT1DM_preprocessed/2018/test/559/', '../data/OhioT1DM_preprocessed/2018/test/563/', '../data/OhioT1DM_preprocessed/2018/test/570/', '../data/OhioT1DM_preprocessed/2018/test/575/', '../data/OhioT1DM_preprocessed/2018/test/588/', '../data/OhioT1DM_preprocessed/2018/test/591/']
paths_test_data_2020 = ['../data/OhioT1DM_preprocessed/2020/test/540/', '../data/OhioT1DM_preprocessed/2020/test/544/', '../data/OhioT1DM_preprocessed/2020/test/552/', '../data/OhioT1DM_preprocessed/2020/test/567/', '../data/OhioT1DM_preprocessed/2020/test/584/', '../data/OhioT1DM_preprocessed/2020/test/596/']


window_size_in_min = 60
prediction_horizon_in_min = 30

########################################### read data #################################################################

# create path to samples
def create_path_to_samples(path):
    path.replace('OhioT1DM_preprocessed', 'OhioT1DM_preprocessed')
    path += '/sample_csvs/window_size_' + str(window_size_in_min) + 'min_prediction_horizon_' + str(prediction_horizon_in_min) + '_min.csv'
    return path

path_to_samples_train_2018 = [create_path_to_samples(path) for path in paths_training_data_2018]
path_to_samples_train_2020 = [create_path_to_samples(path) for path in paths_training_data_2020]
path_to_samples_test_2018 = [create_path_to_samples(path) for path in paths_test_data_2018]
path_to_samples_test_2020 = [create_path_to_samples(path) for path in paths_test_data_2020]

# if needed perform pre-processing
path_to_samples_all = path_to_samples_train_2018 + path_to_samples_train_2020 + path_to_samples_test_2018 + path_to_samples_test_2020
for path in path_to_samples_all:
    if not os.path.exists(path):
        print('pre-processing data because samples do not exist' + path)
        pre_processing.turn_into_samples_all_data(window_size_in_min, prediction_horizon_in_min)
        continue