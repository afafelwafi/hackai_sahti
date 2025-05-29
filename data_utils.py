import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from config_loader import get_config


def find_file_in_folder(folder: str, filename: str) -> str:
    """
    Search for a file in a given folder and its subfolders.
    
    Args:
        folder (str): The root folder to search in.
        filename (str): The name of the file to find.
    
    Returns:
        str: Full path to the file if found, else raises Exception.
    """
    for root, _, files in os.walk(folder):
        if filename in files:
            return os.path.join(root, filename)
    raise Exception(f"File '{filename}' not found in folder '{folder}'.")


def load_dataset(data_dir: str, multi_label: bool = None) -> Tuple[List[str], List, List[str], List[List]]:
    """
    Loads dataset consisting of .wav files and metadata CSVs.

    Args:
        data_dir (str): Directory containing .wav files, train.csv, and additional_metadata.csv.
        multi_label (bool): If True, loads multi-hot encoded labels. Otherwise, only binary class 'N'.
                           If None, uses config default.

    Returns:
        audio_files (List[str]): Paths to audio .wav files.
        label_list (List): Corresponding labels (multi-hot or binary).
        patient_ids (List[str]): Patient identifiers.
        demographic_info (List[List]): Last 4 demographic columns per patient.
    """
    config = get_config()
    
    # Use parameter or config default
    if multi_label is None:
        multi_label = config.multi_label
    
    audio_files = []
    label_list = []
    patient_ids = []
    demographic_info = []

    metadata_path = find_file_in_folder(data_dir, config.train_csv)
    demographics_path = find_file_in_folder(data_dir, config.metadata_csv)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    print(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    demographics = pd.read_csv(demographics_path)

    recording_columns = config.recording_columns

    for root, _, files in os.walk(data_dir):
        for filename in files:
            if not filename.endswith('.wav'):
                continue

            try:
                parts = filename.replace('.wav', '').split('_')
                disease_label, patient_num = parts[0], parts[1]
                patient_id = f"patient_{patient_num}"
            except IndexError:
                print(f"Skipping file with invalid format: {filename}")
                continue

            file_path = os.path.join(root, filename)
            match_patient = metadata['patient_id'] == patient_id
            match_recording = metadata[recording_columns].eq(filename.replace('.wav', '')).any(axis=1)
            matched_rows = metadata[match_patient & match_recording]
            matched_demo = demographics[match_patient]

            if matched_rows.empty:
                print(f"No metadata found for {filename}")
                continue

            row = matched_rows.iloc[0]
            audio_files.append(file_path)
            patient_ids.append(patient_id)

            # Extract label
            if multi_label:
                label = row[config.disease_classes].astype(int).tolist()
            else:
                label = int(row[config.normal_class])
            label_list.append(label)

            # Extract demographic info (last 4 columns)
            demo_row = matched_demo.iloc[0, -4:].tolist()
            demographic_info.append(demo_row)

    return audio_files, label_list, patient_ids, demographic_info


def split_by_subject(
    audio_files: List[str], 
    labels: List, 
    patient_ids: List[str], 
    demographics: List[List], 
    test_ratio: float = None, 
    seed: int = None
) -> Tuple[Tuple, Tuple]:
    """
    Performs subject-level split to avoid patient data leakage and normalizes age.

    Args:
        audio_files (List[str]): Audio file paths.
        labels (List): Corresponding labels.
        patient_ids (List[str]): Patient IDs.
        demographics (List[List]): Demographic data (age should be first element).
        test_ratio (float): Fraction of subjects to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_data, val_data: Tuples of (audio_files, labels, demographics, patient_ids)
    """
    config = get_config()
    
    # Use parameters or config defaults
    if test_ratio is None:
        test_ratio = config.test_ratio
    if seed is None:
        seed = config.random_seed
    
    np.random.seed(seed)

    unique_subjects = list(set(patient_ids))
    num_val = max(1, int(len(unique_subjects) * test_ratio))
    np.random.shuffle(unique_subjects)

    val_subjects = set(unique_subjects[:num_val])
    train_subjects = set(unique_subjects[num_val:])

    train_idx, val_idx = [], []

    for idx, subject in enumerate(patient_ids):
        if subject in train_subjects:
            train_idx.append(idx)
        elif subject in val_subjects:
            val_idx.append(idx)

    def subset(indices):
        return (
            [audio_files[i] for i in indices],
            [labels[i] for i in indices],
            [demographics[i] for i in indices],
            [patient_ids[i] for i in indices]
        )

    train_data = subset(train_idx)
    val_data = subset(val_idx)

    # Normalize age using only train set
    train_demo = train_data[2]
    train_ages = np.array([d[0] for d in train_demo], dtype=np.float32)
    mean_age = np.mean(train_ages)
    std_age = np.std(train_ages) if np.std(train_ages) > 0 else 1.0

    def normalize_age(demo_list):
        normed = []
        for d in demo_list:
            d = d.copy()
            d[0] = (d[0] - mean_age) / std_age
            normed.append(d)
        return normed

    train_data = (
        train_data[0],
        train_data[1],
        normalize_age(train_data[2]),
        train_data[3]
    )
    val_data = (
        val_data[0],
        val_data[1],
        normalize_age(val_data[2]),
        val_data[3]
    )

    print(f"\nTrain subjects: {len(set(train_data[3]))}")
    print(f"Validation subjects: {len(set(val_data[3]))}")
    print(f"Age normalization: mean={mean_age:.2f}, std={std_age:.2f}")

    return train_data, val_data