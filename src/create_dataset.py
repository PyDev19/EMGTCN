from io import BytesIO
from tqdm import tqdm
import zipfile
import requests
import scipy.io as scio
import numpy as np
import os
import pandas as pd

pd.set_option('future.no_silent_downcasting', True)

def download_and_unzip(url, extract_to="data"):
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(extract_to)


for i in tqdm(range(1, 28), desc="Downloading and unzipping datasets"):
    download_and_unzip(
        f"https://ninapro.hevs.ch/files/DB1/Preprocessed/s{i}.zip", extract_to="data"
    )

# Base directories
original_data = "data"  # Raw Ninapro data folder
processed_data = "data/processed/"  # Output folder for structured data

# Number of subjects in Ninapro DB1 (S1 to S27)
subjects = 27

# Indexes for DataFrame columns after merging
REPETITION_COL = 10  # Column index for repetition ID
STIMULUS_COL = 11  # Column index for stimulus (gesture label)
GROUP_COL = 12  # Column index for group (block of gesture trials)

# Create processed folder if it doesn't exist
if not os.path.exists(processed_data):
    os.makedirs(processed_data)

for subject_id in tqdm(range(1, subjects + 1), desc="Processing subjects"):
    for exercise_id in range(1, 4):

        # Load the .mat file for this subject and exercise
        file_path = f"{original_data}/S{subject_id}_A1_E{exercise_id}.mat"
        mat_data = scio.loadmat(file_path)

        # Match length across EMG, repetition, and stimulus arrays
        min_length = np.min(
            [
                mat_data["emg"].shape[0],
                mat_data["rerepetition"].shape[0],
                mat_data["restimulus"].shape[0],
            ]
        )

        # Combine into a single DataFrame (EMG + repetition + stimulus)
        df = pd.DataFrame(
            np.hstack(
                (
                    mat_data["emg"][:min_length, :],  # EMG signals
                    mat_data["rerepetition"][:min_length, :],  # Repetition numbers
                    mat_data["restimulus"][:min_length, :],  # Gesture labels
                )
            )
        )

        # -------------------------------
        # Step 1: Fix repetition for rest periods (0 → previous valid repetition)
        # -------------------------------
        df[REPETITION_COL] = df[REPETITION_COL].replace(to_replace=0, value=None).bfill().ffill().infer_objects(copy=False)
        repetition_ids = df[REPETITION_COL].values.reshape(-1, 1)

        # -------------------------------
        # Step 2: Assign group IDs (block-level grouping)
        # -------------------------------
        df[GROUP_COL] = df[STIMULUS_COL].replace(to_replace=0, value=None).bfill().ffill().infer_objects(copy=False)
        group_ids = df[GROUP_COL].values.reshape(-1, 1)

        # -------------------------------
        # Extract core data fields
        # -------------------------------
        emg_signals = df.loc[:, 0:9].values  # First 10 columns are EMG channels
        gesture_labels = df[STIMULUS_COL].values.reshape(-1, 1)

        # -------------------------------
        # Remove rows with group == 0 (rest without gesture)
        # -------------------------------
        valid_indices = np.squeeze(group_ids != 0)
        emg_signals = emg_signals[valid_indices, :]
        gesture_labels = gesture_labels[valid_indices]
        repetition_ids = repetition_ids[valid_indices]
        group_ids = group_ids[valid_indices]

        # -------------------------------
        # Adjust gesture IDs by exercise (global unique ID)
        # Exercise 1: no shift
        # Exercise 2: +12
        # Exercise 3: +29
        # -------------------------------
        nonzero_gestures = gesture_labels != 0
        if exercise_id == 2:
            gesture_labels[nonzero_gestures] += 12
            group_ids += 12
        elif exercise_id == 3:
            gesture_labels[nonzero_gestures] += 29
            group_ids += 29

        for gesture in np.unique(gesture_labels):
            gesture_mask = np.isin(gesture_labels, gesture)
            gesture_dir = (
                f"{processed_data}/subject-{subject_id:02d}/gesture-{int(gesture)}/rms"
            )
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)

            if gesture == 0:
                for group in np.unique(group_ids):
                    group_mask = np.logical_and(np.isin(group_ids, group), gesture_mask)
    
                    for rep in np.unique(repetition_ids):
                        rep_mask = np.isin(repetition_ids, rep)
                        final_mask = np.squeeze(np.logical_and(group_mask, rep_mask))

                        x = emg_signals[final_mask, :]
                        y = gesture_labels[final_mask]
                        z = group_ids[final_mask]
                        w = repetition_ids[final_mask]

                        scio.savemat(
                            f"{gesture_dir}/rep-{int(rep):02d}_{int(z[0][0]):02d}.mat",
                            {"emg": x, "stimulus": y, "repetition": w, "group": z},
                        )
            else:
                for rep in np.unique(repetition_ids):
                    rep_mask = np.isin(repetition_ids, rep)
                    final_mask = np.squeeze(np.logical_and(gesture_mask, rep_mask))

                    x = emg_signals[final_mask, :]
                    y = gesture_labels[final_mask]
                    z = group_ids[final_mask]
                    w = repetition_ids[final_mask]

                    scio.savemat(
                        f"{gesture_dir}/rep-{int(rep):02d}.mat",
                        {"emg": x, "stimulus": y, "repetition": w, "group": z},
                    )
