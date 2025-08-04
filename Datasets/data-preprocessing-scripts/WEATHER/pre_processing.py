import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from tabulate import tabulate

# -----------------------------
# --- CONFIGURATION & PATHS ---
# -----------------------------
# Path to the CSV file (adjust the path as needed)
DATA_PATH = '../data/Weather/weather_features.csv'

# ADATime save folder
ADATIME_SAVE_FOLDER = "../ADATIME_data/WEATHER"
os.makedirs(ADATIME_SAVE_FOLDER, exist_ok=True)

# Window parameters
WINDOW_SIZE_HOURS = 72   # Input: 3 days (72 hours)
PREDICTION_HOUR = 24     # Label: temperature 24 hours after the end of the window.
# For a window starting at index i:
#   sample = rows [i, i+72) and label = row at index i + 72 + 24 - 1 = i + 95.
# (Because if the window covers hours i to i+71, then the hour at i+95 is 24 hours after hour i+71)

# Define the temperature bins (in Celsius)
# Bins: (<25.5, 25.5-26.5, 26.5-27.5, 27.5-28.5, 28.5-29.5, 29.5-30.5, 30.5-31.5, >=31.5)
temp_bins = [25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5]

readable_labels = ["< 25.5°C", "25.5°C - 26.5°C", "26.5°C - 27.5°C", "27.5°C - 28.5°C", 
                   "28.5°C - 29.5°C", "29.5°C - 30.5°C", "30.5°C - 31.5°C", ">= 31.5°C"]

label_to_range = {
    0: "< 25.5°C",
    1: "25.5°C - 26.5°C",
    2: "26.5°C - 27.5°C",
    3: "27.5°C - 28.5°C",
    4: "28.5°C - 29.5°C",
    5: "29.5°C - 30.5°C",
    6: "30.5°C - 31.5°C",
    7: ">= 31.5°C"
}

# -----------------------------
# --- UTILITY FUNCTIONS ---
# -----------------------------
def discretize_temp(temp):
    """
    Discretize the temperature (in Celsius) into 8 classes.
    Uses np.digitize such that:
        - if temp < 25.5                -> class 0
        - if 25.5 <= temp < 26.5          -> class 1
        - if 26.5 <= temp < 27.5          -> class 2
        - if 27.5 <= temp < 28.5          -> class 3
        - if 28.5 <= temp < 29.5          -> class 4
        - if 29.5 <= temp < 30.5          -> class 5
        - if 30.5 <= temp < 31.5          -> class 6
        - if temp >= 31.5                -> class 7
    """
    return np.digitize(temp, temp_bins, right=False)

def process_city(df, city_name):
    """
    For a given city, sort the data by timestamp, convert temperatures from Kelvin to Celsius,
    and then create sliding window samples and corresponding labels.
    
    Input sample: 72 consecutive hours (all numeric features).
    Label: temperature (converted to Celsius) at 24h after the window (discretized).
    
    Returns:
      samples: NumPy array of shape (n_samples, 72, num_features)
      labels:  NumPy array of shape (n_samples,)
    """
    # Filter the dataframe to the given city
    city_df = df[df["city_name"] == city_name].copy()
    
    # Convert the dt_iso column to datetime with UTC and sort
    city_df["dt_iso"] = pd.to_datetime(city_df["dt_iso"], utc=True)
    city_df.sort_values("dt_iso", inplace=True)
    city_df.reset_index(drop=True, inplace=True)
    
    # Convert temperature columns from Kelvin to Celsius
    for col in ["temp", "temp_min", "temp_max"]:
        city_df[col] = city_df[col] - 273.15
    
    # Select only the numeric features (drop dt_iso, city_name, and non-numeric weather descriptors)
    feature_cols = ["temp", "temp_min", "temp_max", "pressure", "humidity", 
                    "wind_speed", "wind_deg"]
    features = city_df[feature_cols].values

    samples = []
    labels = []
    total_rows = len(city_df)
    # Calculate the last index from which we can extract a sample with a valid label.
    last_valid_idx = total_rows - (WINDOW_SIZE_HOURS + PREDICTION_HOUR - 1)
    
    for i in range(last_valid_idx):
        # Create a sample from i to i+WINDOW_SIZE_HOURS (72 rows)
        window = features[i : i + WINDOW_SIZE_HOURS]
        # The label is the temperature at 24h after the window ends.
        # Since the window covers indices [i, i+71], the label is at index i + 95.
        label_row = city_df.iloc[i + WINDOW_SIZE_HOURS + PREDICTION_HOUR - 1]
        temp_value = label_row["temp"]  # already in Celsius
        label = discretize_temp(temp_value)
        
        samples.append(window)
        labels.append(label)
    
    samples = np.array(samples)  # shape: (n_samples, 72, num_features)
    labels = np.array(labels)    # shape: (n_samples,)
    return samples, labels

# -----------------------------
# --- DATASET PREPROCESSING ---
# -----------------------------
# Load the CSV file
df = pd.read_csv(DATA_PATH)

# List of unique cities (subdomains)
cities = df["city_name"].unique()
print("Found cities:", cities)

# For saving mapping from ADATime file names to city names
train_mapping = []  # List of tuples: (New Filename, Original Name)
test_mapping = []   # List of tuples: (New Filename, Original Name)
train_counter = 0
test_counter = 0

# For each city, create samples and split into train and test sets.
for city in cities:
    print(f"\nProcessing city: {city}")
    samples, labels = process_city(df, city)
    
    n_samples = samples.shape[0]
    if n_samples == 0:
        print(f"Warning: No samples created for city: {city}")
        continue
    
    # Chronological train-test split (first 80% train, last 20% test)
    split_idx = int(0.8 * n_samples)
    train_samples = samples[:split_idx]
    train_labels = labels[:split_idx]
    test_samples = samples[split_idx:]
    test_labels = labels[split_idx:]
    
    # Convert to PyTorch tensors
    train_samples_tensor = torch.tensor(train_samples, dtype=torch.float32)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    test_samples_tensor = torch.tensor(test_samples, dtype=torch.float32)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # Save training data (.pt file)
    train_filename = f"train_{train_counter}.pt"
    torch.save(
        {"samples": train_samples_tensor, "labels": train_labels_tensor},
        os.path.join(ADATIME_SAVE_FOLDER, train_filename)
    )
    # Mapping name format similar to Ohio: domain (city) + original city name
    train_mapping.append((train_filename, city))
    print(f"Saved training file: {train_filename} (n_samples: {train_samples_tensor.shape[0]})")

    # Print additional training data info
    unique_train_labels, train_counts = np.unique(train_labels, return_counts=True)
    print("\n\nTrain Data Details:\n")
    print("  Shape of samples tensor:", train_samples_tensor.shape)
    print("  Shape of labels tensor:", train_labels_tensor.shape)
    print("  Labels:", readable_labels)
    print("  Unique train labels:", unique_train_labels)
    print("  Label counts:", train_counts)

    print("  Train label mapping:")
    for label in unique_train_labels:
        print(f"    {label}: {label_to_range[label]}")

    train_counter += 1

    # Save testing data (.pt file)
    test_filename = f"test_{test_counter}.pt"
    torch.save(
        {"samples": test_samples_tensor, "labels": test_labels_tensor},
        os.path.join(ADATIME_SAVE_FOLDER, test_filename)
    )
    test_mapping.append((test_filename, city))
    print(f"Saved testing file: {test_filename} (n_samples: {test_samples_tensor.shape[0]})")

    # Print additional testing data info
    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
    print("\n\nTest Data Details:\n")
    print("  Shape of samples tensor:", test_samples_tensor.shape)
    print("  Shape of labels tensor:", test_labels_tensor.shape)
    print("  Labels:", readable_labels)
    print("  Unique test labels:", unique_test_labels)
    print("  Label counts:", test_counts)

    print("  Test label mapping:")
    for label in unique_test_labels:
        print(f"    {label}: {label_to_range[label]}")

    test_counter += 1

# -----------------------------
# --- REPORT THE MAPPINGS ---
# -----------------------------
print("\nMapping for training files:")
print(tabulate(train_mapping, headers=["New Filename", "Original Name"], tablefmt="double_grid"))

print("\nMapping for testing files:")
print(tabulate(test_mapping, headers=["New Filename", "Original Name"], tablefmt="double_grid"))
