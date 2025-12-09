"""
data_preprocessing.py (FIXED VERSION)

- Removes numpy.object_ issue
- Ensures all sequences have equal length
- Stores as float32 arrays (not object)
- Loader converts everything safely to torch tensors
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch


# ============================================================
# 1) IMS Dataset Loader (Raw)
# ============================================================

class BearingDataset(Dataset):
    """
    Loads a single raw folder with txt files.
    Applies channel reduction + downsampling.
    """

    def __init__(self, data_dir, downsample_ratio=10):
        self.data_dir = data_dir
        self.file_list = sorted(os.listdir(data_dir))
        self.downsample_ratio = downsample_ratio

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = pd.read_csv(file_path, sep="\t", header=None).values

        # Channel reduction (8 → 4)
        if data.shape[1] == 8:
            data = np.column_stack([
                (data[:, 0] + data[:, 1]) / 2,
                (data[:, 2] + data[:, 3]) / 2,
                (data[:, 4] + data[:, 5]) / 2,
                (data[:, 6] + data[:, 7]) / 2,
            ])

        # Downsampling
        data = data[::self.downsample_ratio]

        return torch.tensor(data, dtype=torch.float32)


# ============================================================
# 2) Save Preprocessed Dataset (FIXED — equal length)
# ============================================================

def preprocess_and_save(data_dir, test_set, save_dir, downsample_ratio=10, limit=None):
    """
    Preprocess directory → produce uniform float32 array.
    Fixes object dtype by forcing equal-length sequences.
    """

    input_dir = os.path.join(data_dir, test_set)
    files = sorted(os.listdir(input_dir))

    if limit:
        files = files[:limit]

    processed = []

    print(f"[Preprocess] Loading {test_set}... (files: {len(files)})")

    # First pass → collect raw sequences
    for file in files:
        try:
            path = os.path.join(input_dir, file)
            data = pd.read_csv(path, sep="\t", header=None).values

            if data.shape[1] == 8:
                data = np.column_stack([
                    (data[:, 0] + data[:, 1]) / 2,
                    (data[:, 2] + data[:, 3]) / 2,
                    (data[:, 4] + data[:, 5]) / 2,
                    (data[:, 6] + data[:, 7]) / 2,
                ])

            data = data[::downsample_ratio]
            processed.append(data)

        except Exception as e:
            print(f"[ERROR] Failed to process file {file}: {e}")

    # Determine minimum length to enforce uniform shape
    min_len = min(seq.shape[0] for seq in processed)

    # Trim and convert to float32 
    processed_fixed = np.array(
        [seq[:min_len].astype(np.float32) for seq in processed],
        dtype=np.float32
    )  # shape: (num_files, min_len, 4)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{test_set}_processed.npy")

    np.save(save_path, processed_fixed)
    print(f"[Saved] Preprocessed dataset: {save_path}")
    print(f" → shape: {processed_fixed.shape}, dtype: {processed_fixed.dtype}")


# ============================================================
# 3) DataLoader (SAFE — no object arrays)
# ============================================================

def create_compressed_dataloaders(npy_file, batch_size=32, shuffle=True, num_workers=0):
    """
    Loads uniform preprocessed data → produces DataLoader
    Shape: (N, seq_len, 4)
    """

    data = np.load(npy_file)  # already float32 & uniform

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )

    return dataloader