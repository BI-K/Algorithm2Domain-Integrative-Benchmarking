 # import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy import signal
import os



# paths to training data
paths_training_data_2018 = ['../data/OhioT1DM/2018/train/559-ws-training.xml', '../data/OhioT1DM/2018/train/563-ws-training.xml', '../data/OhioT1DM/2018/train/570-ws-training.xml', '../data/OhioT1DM/2018/train/575-ws-training.xml', '../data/OhioT1DM/2018/train/588-ws-training.xml', '../data/OhioT1DM/2018/train/591-ws-training.xml']
paths_training_data_2020 =['../data/OhioT1DM/2020/train/540-ws-training.xml', '../data/OhioT1DM/2020/train/544-ws-training.xml', '../data/OhioT1DM/2020/train/552-ws-training.xml', '../data/OhioT1DM/2020/train/567-ws-training.xml', '../data/OhioT1DM/2020/train/584-ws-training.xml', '../data/OhioT1DM/2020/train/596-ws-training.xml']

# path to test data 
path_test_data_2018 = ['../data/OhioT1DM/2018/test/559-ws-testing.xml', '../data/OhioT1DM/2018/test/563-ws-testing.xml', '../data/OhioT1DM/2018/test/570-ws-testing.xml', '../data/OhioT1DM/2018/test/575-ws-testing.xml', '../data/OhioT1DM/2018/test/588-ws-testing.xml', '../data/OhioT1DM/2018/test/591-ws-testing.xml']
path_test_data_2020 = ['../data/OhioT1DM/2020/test/540-ws-testing.xml', '../data/OhioT1DM/2020/test/544-ws-testing.xml', '../data/OhioT1DM/2020/test/552-ws-testing.xml', '../data/OhioT1DM/2020/test/567-ws-testing.xml', '../data/OhioT1DM/2020/test/584-ws-testing.xml', '../data/OhioT1DM/2020/test/596-ws-testing.xml']


# config
max_missing_values_in_series = 3
smoothing_filter = 'savatsky_golay'
smoothing_window = 7

normal_differences_2018_in_s = [60*5]
normal_difference_2018_exception_in_s = []
normal_differences_2020_in_s = [60*5, 60*5 + 1, 60*5 + 2, 60 + 3]
normal_difference_2020_exception_in_s = [10 * 60 + 1]



def check_whether_to_split_here(difference, is_data_from_2018):

    if is_data_from_2018:
         normal_differences_s = normal_differences_2018_in_s
    else:
        normal_differences_s = normal_differences_2020_in_s

    has_normal_difference = False
    for normal_difference_s in normal_differences_s:
        # unwished condition <- is false when this holds for all differences we check for
        if difference.seconds % normal_difference_s != 0 and difference.seconds < normal_difference_s*4 and difference.days < 1:
            has_normal_difference = False or has_normal_difference
            # special exception for 10:01 from 2020 dataset 
            if not(is_data_from_2018) and difference.seconds in normal_difference_2020_exception_in_s:
                has_normal_difference = True or has_normal_difference
            if is_data_from_2018 and difference.seconds in normal_difference_2018_exception_in_s:
                has_normal_difference = True or has_normal_difference
        else:
                has_normal_difference = True or has_normal_difference

    return has_normal_difference

# impute missing values with linear interpolation if less than 5 minutes are missing
def impute_missing_values(data):
    data = data.interpolate(method='linear')
    return data

# read input as np.array
def divide_cgm_signals_and_turn_to_list(path, is_data_from_2018: bool):
    tree = ET.parse(path)
    root = tree.getroot()
    cgm_data = root.find('glucose_level')
    # cut into chunks if missing values more than 3 consecutive emasuremnts are missing
    # get borger indices
    cut_off_indices = [0]
    cgm_chunks = []
    cgm_chunk = [(datetime.strptime(cgm_data[0].attrib['ts'], "%d-%m-%Y %H:%M:%S"), float(cgm_data[0].attrib['value']))]
    for i in range(len(cgm_data)-1):
            #'07-12-2021 01:17:00'
            prev_ts = datetime.strptime(cgm_data[i].attrib['ts'], "%d-%m-%Y %H:%M:%S")
            next_ts = datetime.strptime(cgm_data[i+1].attrib['ts'], "%d-%m-%Y %H:%M:%S")

            difference = next_ts - prev_ts
            if difference.seconds > 5*60:
                if difference.seconds > 5*60 * max_missing_values_in_series or not check_whether_to_split_here(difference, is_data_from_2018):
                    cgm_chunks.append(cgm_chunk)
                    cgm_chunk = [(next_ts, float(cgm_data[i+1].attrib['value']))]
                    cut_off_indices.append(i+1)
                else:
                     # append nan x times
                    for j in range(difference.seconds // (5*60) - 1):
                        prev_ts = prev_ts + pd.Timedelta(5, unit='m')
                        cgm_chunk.append((prev_ts, np.nan)) 
            cgm_chunk.append((next_ts,float(cgm_data[i+1].attrib['value'])))
    
    return cgm_chunks



# smoothing only applied to cgm data
def smooth_cgm_data(cgm_chunks):
    if smoothing_filter == 'savatsky_golay':
        for i in range(len(cgm_chunks)):
            cgm_chunks[i]['cgm'] = signal.savgol_filter(cgm_chunks[i]['glucose_level'], smoothing_window, 2)
            return cgm_chunks
    else:
        return cgm_chunks


# normalization - z-score normalization <- per patient
def normalize_data(cgm_chunks):
    for i in range(len(cgm_chunks)):
        cgm_chunks[i]['glucose_level'] = (cgm_chunks[i]['glucose_level'] - cgm_chunks[i]['glucose_level'].mean()) / cgm_chunks[i]['glucose_level'].std()
    return cgm_chunks

def save_preprocessed_data(path, cgm_chunks):
    # '../data/OhioT1DM/2018/train/559-ws-training.xml'
    path = path.replace('OhioT1DM', 'OhioT1DM_preprocessed')
    path = path.replace('-ws-training.xml', '/').replace('-ws-testing.xml', '/')

    # create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(len(cgm_chunks)):
        cgm_chunks[i].to_csv(path + str(i) +'.csv')
        print("saved file: " + path + str(i) +'.csv')

 
def preprocess_path_cgm(path, is_from_2018):

    print("Preprocessing path: " + path)

    # chunking for cgm data with too many missing values
    cgm_chunks = divide_cgm_signals_and_turn_to_list(path, is_from_2018)
    cgm_chunks = [pd.DataFrame(chunk, columns=['timestamp', 'glucose_level']) for chunk in cgm_chunks]
    cgm_chunks = [chunk.set_index('timestamp') for chunk in cgm_chunks]

    # imputation for few missing values
    cgm_chunks = [impute_missing_values(chunk) for chunk in cgm_chunks]
    cgm_chunks = [chunk.reset_index() for chunk in cgm_chunks]

    # this preprocessing part is done as the authors in A Prior-knowledge-guided Dynamic Attention Mechanism to Predict Nocturnal Hypoglycemic Events in Type 1 Diabetes did
    cgm_chunks = smooth_cgm_data(cgm_chunks)
    cgm_chunks = normalize_data(cgm_chunks)
    
    save_preprocessed_data(path, cgm_chunks)

#for path in path_test_data_2018:
#    preprocess_path_cgm(path, True)

#for path in path_test_data_2020:
#    preprocess_path_cgm(path, False)



# window size and prediction horizon in minutes
def create_ground_truth(time_chunks, window_size, prediction_horizon):
    """
    Create labels for each batch of data window size and prediction horizon in datapoints
    """
    x = []
    for chunk in time_chunks:
        if len(chunk) < window_size + prediction_horizon:
            continue
        for i in range(len(chunk) - window_size - prediction_horizon):
            entry = chunk[i:i+window_size]
            entry.append(chunk[i+window_size+prediction_horizon])
            x.append(entry)


    # trun x into a dataframe, where the last column is labeled as 'prediction'
    columns = [str(i) for i in range(window_size)] + ['prediction']
    x = pd.DataFrame(x, columns=columns)
    return x

def turn_into_samples(path, is_from_2018, window_size_in_min, prediction_horizon_in_min):

    path = path.replace('OhioT1DM', 'OhioT1DM_preprocessed')
    path = path.replace('-ws-training.xml', '/').replace('-ws-testing.xml', '/')
    print("Preprocessing path: " + path)

    # get paths of all csv files in the folder
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    cgm_chunks = [pd.read_csv(path + file)['glucose_level'].tolist() for file in csv_files]
    
    window_size = int(window_size_in_min // 5)
    prediction_horizon = int(prediction_horizon_in_min // 5)
    x= create_ground_truth(cgm_chunks, window_size, prediction_horizon)

    # save preprocessed data
    if not os.path.exists(path + '/sample_csvs/'):
        os.makedirs(path + '/sample_csvs/')
    x.to_csv(path + '/sample_csvs/window_size_' + str(window_size_in_min) + 'min_prediction_horizon_' + str(prediction_horizon_in_min) + '_min.csv')



def turn_into_samples_all_data(window_size_in_min, prediction_horizon_in_min):
    for path in paths_training_data_2018:
        turn_into_samples(path, True, window_size_in_min, prediction_horizon_in_min)

    for path in paths_training_data_2020:
        turn_into_samples(path, False, window_size_in_min, prediction_horizon_in_min)

    for path in path_test_data_2018:
        turn_into_samples(path, True, window_size_in_min, prediction_horizon_in_min)

    for path in path_test_data_2020:
        turn_into_samples(path, False, window_size_in_min, prediction_horizon_in_min)


########################################## Config ####################################################################
# paths to training data
paths_training_data_2018 = [
    "../data/OhioT1DM_preprocessed/2018/train/559/",
    "../data/OhioT1DM_preprocessed/2018/train/563/",
    "../data/OhioT1DM_preprocessed/2018/train/570/",
    "../data/OhioT1DM_preprocessed/2018/train/575/",
    "../data/OhioT1DM_preprocessed/2018/train/588/",
    "../data/OhioT1DM_preprocessed/2018/train/591/",
]
paths_training_data_2020 = [
    "../data/OhioT1DM_preprocessed/2020/train/540/",
    "../data/OhioT1DM_preprocessed/2020/train/544/",
    "../data/OhioT1DM_preprocessed/2020/train/552/",
    "../data/OhioT1DM_preprocessed/2020/train/567/",
    "../data/OhioT1DM_preprocessed/2020/train/584/",
    "../data/OhioT1DM_preprocessed/2020/train/596/",
]

# path to test data
paths_test_data_2018 = [
    "../data/OhioT1DM_preprocessed/2018/test/559/",
    "../data/OhioT1DM_preprocessed/2018/test/563/",
    "../data/OhioT1DM_preprocessed/2018/test/570/",
    "../data/OhioT1DM_preprocessed/2018/test/575/",
    "../data/OhioT1DM_preprocessed/2018/test/588/",
    "../data/OhioT1DM_preprocessed/2018/test/591/",
]
paths_test_data_2020 = [
    "../data/OhioT1DM_preprocessed/2020/test/540/",
    "../data/OhioT1DM_preprocessed/2020/test/544/",
    "../data/OhioT1DM_preprocessed/2020/test/552/",
    "../data/OhioT1DM_preprocessed/2020/test/567/",
    "../data/OhioT1DM_preprocessed/2020/test/584/",
    "../data/OhioT1DM_preprocessed/2020/test/596/",
]


window_size_in_min = 60
prediction_horizon_in_min = 30

########################################### read data #################################################################


# create path to samples
def create_path_to_samples(path):
    path.replace("OhioT1DM_preprocessed", "OhioT1DM_preprocessed")
    path += (
        "/sample_csvs/window_size_"
        + str(window_size_in_min)
        + "min_prediction_horizon_"
        + str(prediction_horizon_in_min)
        + "_min.csv"
    )
    return path


path_to_samples_train_2018 = [
    create_path_to_samples(path) for path in paths_training_data_2018
]
path_to_samples_train_2020 = [
    create_path_to_samples(path) for path in paths_training_data_2020
]
path_to_samples_test_2018 = [
    create_path_to_samples(path) for path in paths_test_data_2018
]
path_to_samples_test_2020 = [
    create_path_to_samples(path) for path in paths_test_data_2020
]

# if needed perform pre-processing
path_to_samples_all = (
    path_to_samples_train_2018
    + path_to_samples_train_2020
    + path_to_samples_test_2018
    + path_to_samples_test_2020
)
for path in path_to_samples_all:
    if not os.path.exists(path):
        print("pre-processing data because samples do not exist" + path)
        turn_into_samples_all_data(
            window_size_in_min, prediction_horizon_in_min
        )
        continue