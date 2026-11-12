"""Load private study inputs and create deterministic train/test datasets.

Labels are control=0 and OTS=1. Tabular rows are sorted by testID.
Sequence files retain historical filesystem traversal order; cross-modality
subject alignment must be verified before using ensemble models. Scalers fit
only the training partition. All loaders use the original 25% test split."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from scipy.signal import decimate
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from torch.utils.data import TensorDataset

from ai_cpet_ots.helpers import normalize, pan_tompkins, trim_spikes
from ai_cpet_ots.paths import require_directory, require_file


class Model_3_Dataset(TensorDataset):
    """Variable-length BBB sequences with optional labels.

    X is a list of (time, features) tensors. Each item is (sequence, label, length);
    unlabeled items retain the historical zero-like placeholder label."""

    def __init__(self, X, Y=None):
        """Initialize Model_3_Dataset with the supplied study configuration."""
        self.lengths = [x.size(0) for x in X]

        self.X = X
        self.Y = Y

    def __len__(self):
        """Return the number of subjects in this dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return the features, labels, and optional length for subject idx."""
        if self.Y is None:
            return self.X[idx], torch.zeros_like(self.X[idx]), self.lengths[idx]
        else:
            return self.X[idx], self.Y[idx], self.lengths[idx]


class Model_4_Dataset(TensorDataset):
    """Variable-length ECG sequences with optional labels.

    X contains one-dimensional sequence tensors. Each item is (sequence, label,
    length); unlabeled items retain the historical zero-like placeholder label."""

    def __init__(self, X, Y=None):
        """Initialize Model_4_Dataset with the supplied study configuration."""
        self.lengths = [x.size(0) for x in X]

        self.X = X
        self.Y = Y

    def __len__(self):
        """Return the number of subjects in this dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return the features, labels, and optional length for subject idx."""
        if self.Y is None:
            return self.X[idx], torch.zeros_like(self.X[idx]), self.lengths[idx]
        else:
            return self.X[idx], self.Y[idx], self.lengths[idx]


class Model_1_2_dataset(TensorDataset):
    """Aligned clinical/CPET arrays exposed as four tensors per subject.

    Items are (clinical_features, clinical_label, CPET_features, CPET_label).
    The caller is responsible for matching subject order across the two arrays."""

    def __init__(self, X_clinical, Y_clinical, X_cpet, Y_cpet):
        """Initialize Model_1_2_dataset with the supplied study configuration."""
        self.X_clinical = torch.tensor(X_clinical)
        self.Y_clinical = torch.tensor(Y_clinical)
        self.X_cpet = torch.tensor(X_cpet)
        self.Y_cpet = torch.tensor(Y_cpet)

    def __len__(self):
        """Return the number of subjects in this dataset."""
        return len(self.X_clinical)

    def __getitem__(self, idx):
        """Return the features, labels, and optional length for subject idx."""
        return self.X_clinical[idx], self.Y_clinical[idx], self.X_cpet[idx], self.Y_cpet[idx]


class Model_1_3_4_dataset(TensorDataset):
    """Aligned clinical, BBB, and ECG samples for the three-model ensemble.

    Items contain features/labels for each modality plus the BBB sequence length.
    Use collate_model_1_3_4 for padding; verify subject alignment before construction."""

    def __init__(self, X_clinical, Y_clinical, X_bbb, Y_bbb, X_ecg, Y_ecg):
        """Initialize Model_1_3_4_dataset with the supplied study configuration."""
        self.lengths = [x.size(0) for x in X_bbb]

        self.X_clinical = torch.tensor(X_clinical)
        self.Y_clinical = torch.tensor(Y_clinical)
        self.X_bbb = X_bbb
        self.Y_bbb = torch.tensor(Y_bbb)
        self.X_ecg = X_ecg
        self.Y_ecg = torch.tensor(Y_ecg)

    def __len__(self):
        """Return the number of subjects in this dataset."""
        return len(self.X_clinical)

    def __getitem__(self, idx):
        """Return the features, labels, and optional length for subject idx."""
        return (
            self.X_clinical[idx],
            self.Y_clinical[idx],
            self.X_bbb[idx],
            self.Y_bbb[idx],
            self.X_ecg[idx],
            self.Y_ecg[idx],
            self.lengths[idx],
        )


def load_clinical_data(control_dir, ots_dir, return_indexes=False):
    # Load data excels
    """Load clinical spreadsheets, encode sex, split subjects, and scale features.

    Inputs accept strings or Paths. Both tables need testID and the four
    clinical columns in study order (age, height, sex, weight). The sex
    encoding is male=1, other=0. Min-max scaling fits training rows only. Returns
    (X, Y, X_train, Y_train, X_test, Y_test); return_indexes appends both index arrays."""
    control_clinical = pd.read_excel(require_file(control_dir))
    ots_clinical = pd.read_excel(require_file(ots_dir))

    control_clinical.sort_values(by="testID", ascending=True, inplace=True)
    ots_clinical.sort_values(by="testID", ascending=True, inplace=True)

    # Remove id column
    control_clinical = control_clinical.drop(columns="testID")
    ots_clinical = ots_clinical.drop(columns="testID")

    # Convert categorical
    control_clinical["sex"] = [
        1 if row == "male" else 0 for row in control_clinical["sex"]
    ]  # 0=woman, 1=man
    ots_clinical["sex"] = [
        1 if row == "male" else 0 for row in ots_clinical["sex"]
    ]  # 0=woman, 1=man

    # Aggregate to X and Y
    X = np.vstack((control_clinical, ots_clinical))

    control_y = np.zeros_like(control_clinical)[:, 0]  # 0=control, 1=OTS
    ots_y = np.ones_like(ots_clinical)[:, 0]  # 0=control, 1=OTS
    Y = np.concatenate((control_y, ots_y))

    # Split to training and testsets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    trainset_idx, testset_idx = next(sss.split(X, Y))
    print(f" Size of trainset {len(trainset_idx)}, size of testset {len(testset_idx)}")
    print(f"Train indeces {trainset_idx[:10]}, test indices {testset_idx[:10]}")

    X_train = X[trainset_idx]
    X_test = X[testset_idx]
    print(X_train.shape, X_test.shape)

    Y_train = Y[trainset_idx]
    Y_test = Y[testset_idx]
    print(Y_train.shape, Y_test.shape)

    # Statistics
    print(f"Age:    {X[:, 0].mean():.3f} ({X[:, 0].std():.3f})")
    print(f"Height: {X[:, 1].mean():.3f} ({X[:, 1].std():.3f})")
    print(
        f"Gender: Women {len(X[np.where(X == 0)[0], 2])} ({(len(X[np.where(X == 0)[0], 2]) / len(X)) * 100:.2f} %), men {len(X[np.where(X == 1)[0], 2])} ({(len(X[np.where(X == 1)[0], 2]) / len(X)) * 100:.2f} %)"
    )
    print(f"Weight: {X[:, 3].mean():.3f} ({X[:, 3].std():.3f})")

    scaler = MinMaxScaler((0, 1))
    scaler = scaler.fit(X_train)

    # Normalize train and testsets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if return_indexes:
        return X, Y, X_train, Y_train, X_test, Y_test, trainset_idx, testset_idx
    else:
        return X, Y, X_train, Y_train, X_test, Y_test


def load_CPET_data(control_dir, ots_dir, return_indexes=False):
    # Load data excels
    """Load the 11 conventional CPET features in their original column order.

    Both spreadsheets need testID. Returns the same six arrays and optional
    split indexes as load_clinical_data; normalization fits training data only."""
    control_cpet = pd.read_excel(require_file(control_dir))
    ots_cpet = pd.read_excel(require_file(ots_dir))

    control_cpet.sort_values(by="testID", ascending=True, inplace=True)
    ots_cpet.sort_values(by="testID", ascending=True, inplace=True)

    # Remove id column
    control_cpet = control_cpet.drop(columns="testID")
    ots_cpet = ots_cpet.drop(columns="testID")

    # Aggregate to X and Y
    X = np.vstack((control_cpet, ots_cpet))

    control_y = np.zeros_like(control_cpet)[:, 0]
    ots_y = np.ones_like(ots_cpet)[:, 0]
    Y = np.concatenate((control_y, ots_y))

    # Split to training and testsets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    trainset_idx, testset_idx = next(sss.split(X, Y))
    print(f" Size of trainset {len(trainset_idx)}, size of testset {len(testset_idx)}")
    print(f"Train indeces {trainset_idx[:10]}, test indices {testset_idx[:10]}")

    X_train = X[trainset_idx]
    X_test = X[testset_idx]
    print(X_train.shape, X_test.shape)

    Y_train = Y[trainset_idx]
    Y_test = Y[testset_idx]
    print(Y_train.shape, Y_test.shape)

    scaler = MinMaxScaler((0, 1))
    scaler = scaler.fit(X_train)

    # Normalize train and testsets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if return_indexes:
        return X, Y, X_train, Y_train, X_test, Y_test, trainset_idx, testset_idx
    else:
        return X, Y, X_train, Y_train, X_test, Y_test


def create_dataset_model_3(control_dir, ots_dir, mode="standard", scale=(0, 1)):
    """Load per-subject BBB workbooks and preprocess one-second sequences.

    Requires testID, time (seconds), and 14 feature columns including VD.
    Irregular sequences use the historical arange(0, final_time) interpolation
    grid; VD values above 10 are converted from millilitres to litres. Spike
    removal precedes train-fitted scaling. Returns X, Y, train/test lists, maximum
    sequence length, and feature names. Files retain filesystem traversal order."""
    X = []
    Y = []
    n_feats = 0
    lengths = []
    colnames = []
    if mode == "minmax":
        scaler = MinMaxScaler(scale)
    elif mode == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    interp_count = 0

    # Control data
    for directory, _, files in os.walk(require_directory(control_dir)):
        for file in files:
            f = pd.read_excel(Path(directory) / file)

            f_interp = f

            # Interpolate irregular recordings to the existing one-second grid.
            if int(f.iloc[-1, 1]) != len(f) - 1:
                f_interp = f.iloc[0:0].copy()
                signal_length = int(f.iloc[-1, 1])

                for feat in range(len(f.columns) - 2):
                    col_data = f.iloc[:, feat + 2]

                    interp_fun = interp1d(f["time"], col_data, fill_value="extrapolate")
                    col_data_interp = interp_fun(np.arange(0, signal_length))

                    f_interp.iloc[:, feat + 2] = col_data_interp
                interp_count += 1

            f = f_interp.drop(["testID", "time"], axis=1)

            # VD unit check
            if (f["VD"] > 10).any():
                f["VD"] = f["VD"] / 1000

            if len(colnames) == 0:
                colnames = f.columns

            if n_feats == 0:
                n_feats = len(f.columns)

            f_tensor = torch.tensor(f.values)
            X.append(f_tensor)
            Y.append(0)
            lengths.append(f_tensor.size(0))

    # OTS data
    for directory, _, files in os.walk(require_directory(ots_dir)):
        for file in files:
            f = pd.read_excel(Path(directory) / file)

            f_interp = f

            # Interpolate irregular recordings to the existing one-second grid.
            if int(f.iloc[-1, 1]) != len(f) - 1:
                f_interp = f.iloc[0:0].copy()
                signal_length = int(f.iloc[-1, 1])

                for feat in range(len(f.columns) - 2):
                    col_data = f.iloc[:, feat + 2]

                    interp_fun = interp1d(f["time"], col_data, fill_value="extrapolate")
                    col_data_interp = interp_fun(np.arange(0, signal_length))

                    f_interp.iloc[:, feat + 2] = col_data_interp
                interp_count += 1

            f = f_interp.drop(["testID", "time"], axis=1)

            # VD unit check
            if (f["VD"] > 10).any():
                f["VD"] = f["VD"] / 1000

            f_tensor = torch.tensor(f.values)
            X.append(f_tensor)
            Y.append(1)
            lengths.append(f_tensor.size(0))

    # Preprocess the data - replace values that exceed [-3sd : 3sd]
    for x in X:
        for f in range(x.size(1)):
            feat_data = x[:, f]

            window_size = 90
            feat_data_med = median_filter(feat_data, window_size)
            feat_error = feat_data - feat_data_med

            std = feat_error.std()
            n_vals_to_corr = len(feat_error[feat_error > 3 * std]) + len(
                feat_error[feat_error < -3 * std]
            )

            while n_vals_to_corr > 0:
                feat_error[feat_error > 3 * std] = 0
                feat_error[feat_error < -3 * std] = 0
                std = feat_error.std()
                n_vals_to_corr = len(feat_error[feat_error > 3 * std]) + len(
                    feat_error[feat_error < -3 * std]
                )

            x[:, f] = feat_error + feat_data_med

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    trainset_idx, testset_idx = next(sss.split(X, Y))
    print(f"Train indeces {trainset_idx[:10]}, test indices {testset_idx[:10]}")

    X_train = [X[i] for i in trainset_idx]
    Y_train = [Y[i] for i in trainset_idx]
    X_test = [X[i] for i in testset_idx]
    Y_test = [Y[i] for i in testset_idx]
    print(f"Trainset {len(trainset_idx)}, testset {len(testset_idx)}")
    print(
        f"Length mean {np.mean(lengths):.2f} ({np.std(lengths):.2f}) - max/min {np.max(lengths)}/{np.min(lengths)}"
    )
    print(f"Samples interpolated {interp_count}")

    # Reshape X_Train featurewise to compute standardization coefficients
    X_train_featwise = torch.empty((0, n_feats))
    for row in X_train:
        X_train_featwise = torch.vstack((X_train_featwise, row))

    # Fit scaler
    scaler = scaler.fit(X_train_featwise)

    for i, feat in enumerate(colnames):
        print(
            f"{feat} mean {X_train_featwise[:, i].mean():.2f} ({X_train_featwise[:, i].std():.2f}) - max/min {X_train_featwise[:, i].max(dim=0).values:.2f}/{X_train_featwise[:, i].min(dim=0).values:.2f}"
        )

    # Transform train
    for i, row in enumerate(X_train):
        X_train[i] = torch.tensor(scaler.transform(row))
    for i, row in enumerate(X_test):
        X_test[i] = torch.tensor(scaler.transform(row))

    max_length = np.max(lengths)

    return X, Y, X_train, Y_train, X_test, Y_test, max_length, colnames


def create_dataset_model_4(control_dir, ots_dir, filter_mode=None):
    """Load trimmed ECG arrays, decimate from 1000 to 500 Hz, and normalize.

    Reads the first stored channel using the historical f[0, :].transpose()
    selection, skips NaN/high-mean inputs, applies the study filters, and splits
    subjects. Returns X_train, Y_train, X_test, Y_test. filter_mode is retained
    for compatibility; the historical filtering pipeline runs for every value."""
    X = []
    Y = []
    i = 0

    print(f"Filter mode {filter_mode}")

    # Control data
    for directory, _, files in os.walk(require_directory(control_dir)):
        for file in files:
            f = np.load(Path(directory) / file)

            if i < 5:
                i += 1

            if np.isnan(f[0, :]).any() or (f[0, :].mean() > 10).any():
                print(f"Found error in {file} - file skipped")
            else:
                f = f[0, :].transpose()  # (seq_len, channels)
                f = decimate(f, 2, axis=0)  # Downsample to fs=500Hz
                f = pan_tompkins(f)
                f = trim_spikes(f)
                f = normalize(f)
                f = f.astype(np.float32)

                X.append(torch.tensor(f))
                Y.append(0)

    print("Control done")

    i = 0

    # OTS data
    for directory, _, files in os.walk(require_directory(ots_dir)):
        for file in files:
            f = np.load(Path(directory) / file)

            if i < 5:
                i += 1

            if np.isnan(f[0, :]).any() or (f[0, :].mean() > 10).any():
                print(f"Found error in {file} - file skipped")
            else:
                f = f[0, :].transpose()  # (seq_len, channels)
                f = decimate(f, 2, axis=0)  # Downsample to fs=500Hz
                f = pan_tompkins(f)
                f = trim_spikes(f)
                f = normalize(f)
                f = f.astype(np.float32)

                X.append(torch.tensor(f))
                Y.append(1)

    print("OTS done")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    trainset_idx, testset_idx = next(sss.split(X, Y))

    X_train = [X[i] for i in trainset_idx]
    Y_train = [Y[i] for i in trainset_idx]
    X_test = [X[i] for i in testset_idx]
    Y_test = [Y[i] for i in testset_idx]
    print(f"Trainset {len(trainset_idx)}, testset {len(testset_idx)}")

    return X_train, Y_train, X_test, Y_test
