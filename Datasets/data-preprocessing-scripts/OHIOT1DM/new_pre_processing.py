# %%
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tabulate import tabulate
import xml.etree.ElementTree as ET
from datetime import datetime
from scipy import signal

# %%
# ==============================
# --- CONFIGURATION & PATHS ---
# ==============================
# Data file paths
paths_training_data_2018 = [
    "../data/OhioT1DM/2018/train/559-ws-training.xml",
    "../data/OhioT1DM/2018/train/563-ws-training.xml",
    "../data/OhioT1DM/2018/train/570-ws-training.xml",
    "../data/OhioT1DM/2018/train/575-ws-training.xml",
    "../data/OhioT1DM/2018/train/588-ws-training.xml",
    "../data/OhioT1DM/2018/train/591-ws-training.xml",
]
paths_training_data_2020 = [
    "../data/OhioT1DM/2020/train/540-ws-training.xml",
    "../data/OhioT1DM/2020/train/544-ws-training.xml",
    "../data/OhioT1DM/2020/train/552-ws-training.xml",
    "../data/OhioT1DM/2020/train/567-ws-training.xml",
    "../data/OhioT1DM/2020/train/584-ws-training.xml",
    "../data/OhioT1DM/2020/train/596-ws-training.xml",
]
path_test_data_2018 = [
    "../data/OhioT1DM/2018/test/559-ws-testing.xml",
    "../data/OhioT1DM/2018/test/563-ws-testing.xml",
    "../data/OhioT1DM/2018/test/570-ws-testing.xml",
    "../data/OhioT1DM/2018/test/575-ws-testing.xml",
    "../data/OhioT1DM/2018/test/588-ws-testing.xml",
    "../data/OhioT1DM/2018/test/591-ws-testing.xml",
]
path_test_data_2020 = [
    "../data/OhioT1DM/2020/test/540-ws-testing.xml",
    "../data/OhioT1DM/2020/test/544-ws-testing.xml",
    "../data/OhioT1DM/2020/test/552-ws-testing.xml",
    "../data/OhioT1DM/2020/test/567-ws-testing.xml",
    "../data/OhioT1DM/2020/test/584-ws-testing.xml",
    "../data/OhioT1DM/2020/test/596-ws-testing.xml",
]

# paths to training data
p_training_data_2018 = [
    "../data/OhioT1DM_preprocessed/2018/train/559/",
    "../data/OhioT1DM_preprocessed/2018/train/563/",
    "../data/OhioT1DM_preprocessed/2018/train/570/",
    "../data/OhioT1DM_preprocessed/2018/train/575/",
    "../data/OhioT1DM_preprocessed/2018/train/588/",
    "../data/OhioT1DM_preprocessed/2018/train/591/",
]
p_training_data_2020 = [
    "../data/OhioT1DM_preprocessed/2020/train/540/",
    "../data/OhioT1DM_preprocessed/2020/train/544/",
    "../data/OhioT1DM_preprocessed/2020/train/552/",
    "../data/OhioT1DM_preprocessed/2020/train/567/",
    "../data/OhioT1DM_preprocessed/2020/train/584/",
    "../data/OhioT1DM_preprocessed/2020/train/596/",
]

# path to test data
p_test_data_2018 = [
    "../data/OhioT1DM_preprocessed/2018/test/559/",
    "../data/OhioT1DM_preprocessed/2018/test/563/",
    "../data/OhioT1DM_preprocessed/2018/test/570/",
    "../data/OhioT1DM_preprocessed/2018/test/575/",
    "../data/OhioT1DM_preprocessed/2018/test/588/",
    "../data/OhioT1DM_preprocessed/2018/test/591/",
]
p_test_data_2020 = [
    "../data/OhioT1DM_preprocessed/2020/test/540/",
    "../data/OhioT1DM_preprocessed/2020/test/544/",
    "../data/OhioT1DM_preprocessed/2020/test/552/",
    "../data/OhioT1DM_preprocessed/2020/test/567/",
    "../data/OhioT1DM_preprocessed/2020/test/584/",
    "../data/OhioT1DM_preprocessed/2020/test/596/",
]

# %%
# Preprocessing parameters
max_missing_values_in_series = 3
smoothing_filter = "savatsky_golay"
smoothing_window = 7

# Timing differences (in seconds) that indicate regular measurements.
normal_differences_2018_in_s = [60 * 5]
normal_difference_2018_exception_in_s = []
normal_differences_2020_in_s = [60 * 5, 60 * 5 + 1, 60 * 5 + 2, 60 + 3]
normal_difference_2020_exception_in_s = [10 * 60 + 1]

# AdaTime sample configuration (in minutes)
WINDOW_SIZE_MIN = 60       # 60 minutes window
PREDICTION_HORIZON_MIN = 30  # predict 30 minutes ahead
HYPO_THRESHOLD = 90        # hypoglycemia threshold (mg/dL)
# Folder where processed data will be saved
ADATIME_SAVE_FOLDER = "../ADATIME_data/OHIO"

# %%
# ==============================
# --- UTILITY FUNCTIONS ---
# ==============================
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


def impute_missing_values(data):
    # Linear interpolation for short missing gaps
    return data.interpolate(method="linear")


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


def smooth_cgm_data(cgm_chunks):
    if smoothing_filter == 'savatsky_golay':
        for i in range(len(cgm_chunks)):
            cgm_chunks[i]['cgm'] = signal.savgol_filter(cgm_chunks[i]['glucose_level'], smoothing_window, 2)
            return cgm_chunks
    else:
        return cgm_chunks


def normalize_data(cgm_chunks):
    for i in range(len(cgm_chunks)):
        cgm_chunks[i]['glucose_level'] = (cgm_chunks[i]['glucose_level'] - cgm_chunks[i]['glucose_level'].mean()) / cgm_chunks[i]['glucose_level'].std()
    return cgm_chunks

# %%
# ==============================
# --- DATASET PREPROCESSING ---
# ==============================

# Global counters and mapping lists
train_counter = 0
test_counter = 0
train_mapping = []  # List of tuples: (new_filename, original_name)
test_mapping = []   # List of tuples: (new_filename, original_name)

def save_preprocessed_data(path, cgm_chunks):
    # '../data/OhioT1DM/2018/train/559-ws-training.xml'
    path = path.replace('OhioT1DM', 'OhioT1DM_preprocessed')
    path = path.replace('-ws-training.xml', '/').replace('-ws-testing.xml', '/')

    # create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(len(cgm_chunks)):
        cgm_chunks[i].to_csv(path + str(i) +'.csv', index=False)
        print("saved file: " + path + str(i) +'.csv')

def process_file(path, is_from_2018, window_size_in_min, prediction_horizon_in_min, hypo_threshold):
    print("Processing file: " + path)
    # Chunk the raw data
    cgm_chunks = divide_cgm_signals_and_turn_to_list(path, is_from_2018)
    cgm_chunks = [pd.DataFrame(chunk, columns=['timestamp', 'glucose_level']) for chunk in cgm_chunks]
    cgm_chunks = [chunk.set_index('timestamp') for chunk in cgm_chunks]

    # Impute missing values and reset index
    cgm_chunks = [impute_missing_values(chunk) for chunk in cgm_chunks]
    cgm_chunks = [chunk.reset_index() for chunk in cgm_chunks]
    
    # Smoothing (applied on raw values)
    cgm_chunks = smooth_cgm_data(cgm_chunks)
    
    # Save a copy of the smoothed (but not yet normalized) data for label computation
    raw_chunks = [chunk.copy() for chunk in cgm_chunks]
    
    # Normalization (applied on chunks, not per window)
    cgm_chunks = normalize_data(cgm_chunks)

    # Save the preprocessed data
    save_preprocessed_data(path, cgm_chunks)

    path = path.replace('OhioT1DM', 'OhioT1DM_preprocessed')
    path = path.replace('-ws-training.xml', '/').replace('-ws-testing.xml', '/')

    # 1. Read CSV files in sorted order. Then, extract glucose_level column from each CSV file.
    # Sort the CSV file names to ensure consistent order
    # csv_files = sorted(
    #     [f for f in os.listdir(path) if f.endswith('.csv')],
    #     key=lambda x: int(x.split('.')[0])
    # )
    # cgm_chunks = [pd.read_csv(os.path.join(path, file))['glucose_level'].to_list() for file in csv_files]

    # 2. Read CSV files in the directory and extract glucose_level column from each CSV file.
    # Ensure the raw_chunks are in the same order as the CSV files
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    cgm_chunks = [pd.read_csv(path + file)['glucose_level'].to_list() for file in csv_files]

    # Reorder raw_chunks to match the order of csv_files (based on their numeric prefix)
    ordered_raw_chunks = []
    for file in csv_files:
        idx = int(file.split('.')[0])
        ordered_raw_chunks.append(raw_chunks[idx])
    raw_chunks = ordered_raw_chunks

    # Print number of normalized chunks and their lengths
    print("Number of normalized chunks:\t", len(cgm_chunks))
    print("Lengths of normalized chunks:\t", [len(chunk) for chunk in cgm_chunks])
    print("Number of raw chunks:\t\t", len(raw_chunks))
    print("Lengths of raw chunks:\t\t", [len(chunk) for chunk in raw_chunks])
    print("CSV files in the directory:\t", csv_files)

    # Set up sliding window parameters
    window_size = int(window_size_in_min // 5)
    prediction_horizon = int(prediction_horizon_in_min // 5)
    samples = []
    labels = []

    j = 0
    
    # Create sliding windows: use normalized values for features, but compute labels from raw values
    for norm_chunk, raw_chunk in zip(cgm_chunks, raw_chunks):
        # Normalized values for features
        feature_values = norm_chunk
        # Raw values for label computation
        raw_values = raw_chunk["glucose_level"].tolist()
        
        if len(raw_values) < window_size + prediction_horizon:
            continue
        for i in range(len(raw_values) - window_size - prediction_horizon):
            # Use normalized values to create the feature window
            window = feature_values[i : i + window_size]
            if j == 0:
                print("Window shape:", np.array(window).shape)
                print("Window values:", window)
                j += 1
            # Compute label using the raw glucose value at the prediction time
            pred_val = raw_values[i + window_size + prediction_horizon]
            label = 1 if pred_val < hypo_threshold else 0
            
            samples.append(window)
            labels.append(label)
    
    if len(samples) == 0:
        print("Warning: No valid samples created for file:", path)
        return None, None
    
    samples_tensor = torch.tensor(samples, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Save the combined samples and labels as a .csv file
    new_df = pd.DataFrame(samples)
    new_df["label"] = labels
    num_in_file = path.split("/")[-1].split("-")[0]
    t_t = "train" if "train" in path else "test"
    if not os.path.exists("sample_csvs"):
        os.makedirs("sample_csvs")
    new_df.to_csv(f"sample_csvs/WS60_MPH_30_FILE_{t_t}_{num_in_file}.csv", index=False)

    print(f"----- Processed file: {num_in_file} -----")
    print("Length of samples:", len(samples))
    print("Length of labels:", len(labels))
    
    return samples_tensor, labels_tensor


def process_dataset(paths, is_from_2018, window_size_in_min, prediction_horizon_in_min, hypo_threshold, domain):
    """
    Process a list of XML files and save each one as a .pt file in the AdaTime format.
    The file is saved under ADATIME_SAVE_FOLDER/domain_{idx}.pt.
    """
    global train_counter, test_counter, train_mapping, test_mapping
    os.makedirs(ADATIME_SAVE_FOLDER, exist_ok=True)
    for idx, path in enumerate(paths):
        samples, labels = process_file(path, is_from_2018, window_size_in_min, prediction_horizon_in_min, hypo_threshold)
        if samples is None or labels is None:
            continue
        num_in_file = path.split("/")[-1].split("-")[0]
        # Determine if this is a training or testing file based on the domain string
        if "train" in domain:
            new_filename = f"train_{train_counter}.pt"
            # Record the mapping from the new filename to the original (e.g., "train_2018_559")
            train_mapping.append((new_filename, f"{domain}_{num_in_file}"))
            train_counter += 1
        else:
            new_filename = f"test_{test_counter}.pt"
            test_mapping.append((new_filename, f"{domain}_{num_in_file}"))
            test_counter += 1
        save_path = os.path.join(ADATIME_SAVE_FOLDER, new_filename)
        torch.save({'samples': samples, 'labels': labels}, save_path)
        print("Saved:", save_path)

# %%
# ==============================
# --- RUN THE PREPROCESSING ---
# ==============================
# Process training and test files from both years
process_dataset(paths_training_data_2018, True, WINDOW_SIZE_MIN, PREDICTION_HORIZON_MIN, HYPO_THRESHOLD, domain="train_2018")
process_dataset(paths_training_data_2020, False, WINDOW_SIZE_MIN, PREDICTION_HORIZON_MIN, HYPO_THRESHOLD, domain="train_2020")
process_dataset(path_test_data_2018, True, WINDOW_SIZE_MIN, PREDICTION_HORIZON_MIN, HYPO_THRESHOLD, domain="test_2018")
process_dataset(path_test_data_2020, False, WINDOW_SIZE_MIN, PREDICTION_HORIZON_MIN, HYPO_THRESHOLD, domain="test_2020")

# After processing, print the mapping tables
print("\nMapping for training files:")
print(tabulate(train_mapping, headers=["New Filename", "Original Name"], tablefmt="double_grid"))
print("\nMapping for testing files:")
print(tabulate(test_mapping, headers=["New Filename", "Original Name"], tablefmt="double_grid"))

# %%
# Count the unique labels in one of the processed files
print("\n---------------------------")
print("----- Train 2018 559 ------")
print("---------------------------\n")
data = torch.load(os.path.join(ADATIME_SAVE_FOLDER, "train_0.pt"))
unique, counts = torch.unique(data['labels'], return_counts=True)
print("Unique labels:", unique)
print("Counts:", counts)

# View the one of the processed files
print("Samples shape:", data['samples'].shape)
print("Labels shape:", data['labels'].shape)
print("Example sample:", data['samples'][0])
print("Example label:", data['labels'][0])

# %%
print("\n---------------------------")
print("----- Test 2018 559 ------")
print("---------------------------\n")
# Count the unique labels in one of the processed files
data = torch.load(os.path.join(ADATIME_SAVE_FOLDER, "test_0.pt"))
unique, counts = torch.unique(data['labels'], return_counts=True)
print("Unique labels:", unique)
print("Counts:", counts)

# View the one of the processed files
print("Samples shape:", data['samples'].shape)
print("Labels shape:", data['labels'].shape)
print("Example sample:", data['samples'][0])
print("Example label:", data['labels'][0])

# %%



