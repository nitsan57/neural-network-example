import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
import warnings
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# =====================================
# DATA GENERATION
# =====================================


def generate_synthetic_data():
    """Generate synthetic electric board sensor data"""
    print("Generating synthetic electric board data...")

    # Parameters
    n_devices = 40
    n_days = 30
    samples_per_day = 24  # hourly samples
    n_samples = n_days * samples_per_day

    all_data = []

    for device_id in range(n_devices):
        # Create time index
        start_date = datetime(2024, 1, 1) + timedelta(days=device_id % 30)
        time_index = pd.date_range(start=start_date, periods=n_samples, freq="H")

        # Base patterns with some device-specific variations
        device_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2

        # Generate base signals with daily and weekly patterns
        t = np.arange(n_samples)
        daily_pattern = np.sin(2 * np.pi * t / 24)  # Daily cycle
        weekly_pattern = 0.3 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly cycle

        # Current signals (with correlation)
        base_current = 8 + 4 * daily_pattern + 2 * weekly_pattern
        current_1 = base_current * device_factor + np.random.normal(0, 1, n_samples)
        current_2 = base_current * device_factor * 0.9 + np.random.normal(
            0, 1.2, n_samples
        )

        # Clip to reasonable ranges
        current_1 = np.clip(current_1, 2, 35)
        current_2 = np.clip(current_2, 2, 35)

        # Temperature signals (correlated with current)
        temp_base = 25 + 8 * daily_pattern + 3 * weekly_pattern
        temp_1 = temp_base + 0.3 * current_1 + np.random.normal(0, 2, n_samples)
        temp_2 = temp_base + 0.25 * current_2 + np.random.normal(0, 2.5, n_samples)
        temp_3 = (
            temp_base
            + 0.2 * (current_1 + current_2)
            + np.random.normal(0, 2, n_samples)
        )

        # Clip temperatures
        temp_1 = np.clip(temp_1, 15, 60)
        temp_2 = np.clip(temp_2, 15, 60)
        temp_3 = np.clip(temp_3, 15, 60)

        # Fan speeds (inversely correlated with temperature)
        fan_base = 2000 + 800 * daily_pattern
        fan_speed_1 = (
            fan_base + 15 * (temp_1 - 30) + np.random.normal(0, 100, n_samples)
        )
        fan_speed_2 = (
            fan_base + 12 * (temp_2 - 30) + np.random.normal(0, 120, n_samples)
        )

        # Clip fan speeds
        fan_speed_1 = np.clip(fan_speed_1, 1000, 4000)
        fan_speed_2 = np.clip(fan_speed_2, 1000, 4000)

        # Create device dataframe
        device_data = pd.DataFrame(
            {
                "device_id": device_id,
                "timestamp": time_index,
                "current_1[A]": current_1,
                "current_2[A]": current_2,
                "temp_1[c]": temp_1,
                "temp_2[c]": temp_2,
                "temp_3[c]": temp_3,
                "fan_speed_1[rpm]": fan_speed_1,
                "fan_speed_2[rpm]": fan_speed_2,
            }
        )

        all_data.append(device_data)

    # Combine all devices
    df = pd.concat(all_data)
    print(f"Generated data shape: {df.shape}")
    df = create_labels(df)
    print(f"After Lables: generated data shape: {df.shape}")
    return df


def create_labels(df):
    """Create regression and classification labels"""
    print("Creating labels...")

    # Regression target: received signal power (linear combination of currents)
    df["y_received_signal[dBm]"] = (
        -50
        + 2 * df["current_1[A]"]
        + 1.5 * df["current_2[A]"]
        + np.random.normal(0, 2, len(df))
    )

    # Classification target: malfunction detection
    malfunction = np.zeros(len(df), dtype=bool)

    # Group by device to check consecutive conditions
    for device_id in df["device_id"].unique():
        device_mask = df["device_id"] == device_id
        device_indices = df[device_mask].index

        # Check conditions for this device
        high_temp = (
            (df.loc[device_mask, "temp_1[c]"] > 40)
            | (df.loc[device_mask, "temp_2[c]"] > 40)
            | (df.loc[device_mask, "temp_3[c]"] > 40)
        )
        high_current = (df.loc[device_mask, "current_1[A]"] > 20) | (
            df.loc[device_mask, "current_2[A]"] > 20
        )

        condition = high_temp | high_current
        condition_array = condition.values

        # Find 3+ consecutive True values
        for i in range(len(condition_array) - 2):
            if condition_array[i] and condition_array[i + 1] and condition_array[i + 2]:
                # Mark these and subsequent points as malfunction
                malfunction[device_indices[i : min(i + 5, len(device_indices))]] = True

    df["y_malfunction"] = malfunction

    print(f"Malfunction rate: {malfunction.mean():.3f}")
    print(
        f"Signal power range: [{df['y_received_signal[dBm]'].min():.1f}, {df['y_received_signal[dBm]'].max():.1f}]"
    )

    return df


# =====================================
# DATA PREPROCESSING
# =====================================


def plot_sample_data(df):
    """Plot head of dataframe for one feature"""
    print("\n0. Plotting sample data...")

    # Get first device data
    device_0 = df[df["device_id"] == 0].head(48)  # First 2 days
    device_0.set_index("timestamp", inplace=True)
    feature_cols = [
        "current_1[A]",
        "current_2[A]",
        "temp_1[c]",
        "temp_2[c]",
        "temp_3[c]",
        "fan_speed_1[rpm]",
        "fan_speed_2[rpm]",
    ]
    n_features = len(feature_cols)
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 2.5 * n_features), sharex=True)
    for i, c in enumerate(feature_cols):
        axes[i].plot(device_0.index, device_0[c], label=c)
        axes[i].set_title(f"Device 0 - {c} over time")
        axes[i].set_ylabel(c)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    plt.show()
    print("\nFirst 10 rows of the dataset:")
    print(
        df.head(10)[
            [
                "device_id",
                "timestamp",
                "current_1[A]",
                "temp_1[c]",
                "y_received_signal[dBm]",
                "y_malfunction",
            ]
        ]
    )


def resample_and_interpolate(df):
    """Resample to 1 hour and interpolate missing data"""
    print("\n1. Resampling and interpolating data...")
    df_resampled = df.set_index("timestamp").sort_index()
    df_resampled = df_resampled.resample("1H").mean()
    print(f"After resampling: {df_resampled.shape}")
    df_resampled = df_resampled.interpolate(method="linear", limit=2)

    return df_resampled


def split_data(df):
    """Split data into train, validation, and test sets"""
    print("\n2. Splitting data...")

    # Split by devices to avoid data leakage
    device_ids = df["device_id"].unique()

    # 70% train, 15% validation, 15% test
    train_devices, temp_devices = train_test_split(
        device_ids, test_size=0.3, random_state=42
    )
    val_devices, test_devices = train_test_split(
        temp_devices, test_size=0.5, random_state=42
    )

    train_df = df[df["device_id"].isin(train_devices)].copy()
    val_df = df[df["device_id"].isin(val_devices)].copy()
    test_df = df[df["device_id"].isin(test_devices)].copy()

    print(f"Train: {len(train_devices)} devices, {len(train_df)} samples")
    print(f"Validation: {len(val_devices)} devices, {len(val_df)} samples")
    print(f"Test: {len(test_devices)} devices, {len(test_df)} samples")

    return train_df, val_df, test_df


def normalize_data(train_df, val_df, test_df):
    """Calculate z-score parameters on train data and apply to all sets"""
    print("\n3. Normalizing data...")

    feature_cols = [
        "current_1[A]",
        "current_2[A]",
        "temp_1[c]",
        "temp_2[c]",
        "temp_3[c]",
        "fan_speed_1[rpm]",
        "fan_speed_2[rpm]",
    ]

    # Fit scaler on training data
    scaler_X = StandardScaler()
    scaler_y_reg = StandardScaler()

    # Fit on training data
    train_X = train_df[feature_cols].values
    train_y_reg = train_df[["y_received_signal[dBm]"]].values

    scaler_X.fit(train_X)
    scaler_y_reg.fit(train_y_reg)

    # Transform all datasets
    train_df_norm = train_df.copy()
    val_df_norm = val_df.copy()
    test_df_norm = test_df.copy()

    train_df_norm[feature_cols] = scaler_X.transform(train_df[feature_cols])
    val_df_norm[feature_cols] = scaler_X.transform(val_df[feature_cols])
    test_df_norm[feature_cols] = scaler_X.transform(test_df[feature_cols])

    train_df_norm["y_received_signal[dBm]"] = scaler_y_reg.transform(
        train_df[["y_received_signal[dBm]"]]
    ).flatten()
    val_df_norm["y_received_signal[dBm]"] = scaler_y_reg.transform(
        val_df[["y_received_signal[dBm]"]]
    ).flatten()
    test_df_norm["y_received_signal[dBm]"] = scaler_y_reg.transform(
        test_df[["y_received_signal[dBm]"]]
    ).flatten()

    print("Normalization completed.")
    print(f"Feature means: {scaler_X.mean_}")
    print(f"Feature stds: {scaler_X.scale_}")

    return train_df_norm, val_df_norm, test_df_norm, scaler_X, scaler_y_reg


# =====================================
# DATASET AND MODEL CLASSES
# =====================================


class ElectricBoardDataset(Dataset):
    def __init__(self, df, sequence_length=24, task="regression"):
        self.df = df.sort_values(["device_id", "timestamp"]).reset_index(drop=True)
        self.sequence_length = sequence_length
        self.task = task
        self.feature_cols = [
            "current_1[A]",
            "current_2[A]",
            "temp_1[c]",
            "temp_2[c]",
            "temp_3[c]",
            "fan_speed_1[rpm]",
            "fan_speed_2[rpm]",
        ]

        # Create valid sequences
        self.valid_indices = []
        for device_id in self.df["device_id"].unique():
            device_data = self.df[self.df["device_id"] == device_id]
            device_indices = device_data.index.tolist()

            # Create sequences of length sequence_length
            for i in range(len(device_indices) - sequence_length + 1):
                self.valid_indices.append(device_indices[i : i + sequence_length])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        indices = self.valid_indices[idx]

        # Get sequence data
        X = self.df.loc[indices, self.feature_cols].values.astype(np.float32)

        # Get target (last timestep)
        if self.task == "regression":
            y = self.df.loc[indices[-1], "y_received_signal[dBm]"].astype(np.float32)
        else:  # classification
            y = self.df.loc[indices[-1], "y_malfunction"].astype(np.float32)

        return torch.tensor(X), torch.tensor(y)


class RNNModel(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, task="regression"):
        super(RNNModel, self).__init__()
        self.task = task
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Identity()  # Default activation
        if self.task == "classification":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Take the last output
        last_output = lstm_out[:, -1, :]

        # Final prediction
        output = self.fc(last_output)

        output = self.activation(output)

        return output.squeeze()


def get_data_loaders_reg(
    train_df_norm, val_df_norm, test_df_norm, sequence_length, batch_size
):
    # Regression datasets
    train_dataset_reg = ElectricBoardDataset(
        train_df_norm, sequence_length, "regression"
    )
    val_dataset_reg = ElectricBoardDataset(val_df_norm, sequence_length, "regression")
    test_dataset_reg = ElectricBoardDataset(test_df_norm, sequence_length, "regression")

    train_loader_reg = DataLoader(
        train_dataset_reg, batch_size=batch_size, shuffle=True
    )
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=batch_size, shuffle=False)
    test_loader_reg = DataLoader(test_dataset_reg, batch_size=batch_size, shuffle=False)
    return (
        train_loader_reg,
        val_loader_reg,
        test_loader_reg,
    )


def get_classification_data_loaders(
    train_df_norm, val_df_norm, test_df_norm, sequence_length, batch_size
):
    # Classification datasets
    train_dataset_cls = ElectricBoardDataset(
        train_df_norm, sequence_length, "classification"
    )
    val_dataset_cls = ElectricBoardDataset(
        val_df_norm, sequence_length, "classification"
    )
    test_dataset_cls = ElectricBoardDataset(
        test_df_norm, sequence_length, "classification"
    )

    train_loader_cls = DataLoader(
        train_dataset_cls, batch_size=batch_size, shuffle=True
    )
    val_loader_cls = DataLoader(val_dataset_cls, batch_size=batch_size, shuffle=False)
    test_loader_cls = DataLoader(test_dataset_cls, batch_size=batch_size, shuffle=False)
    return (
        train_loader_cls,
        val_loader_cls,
        test_loader_cls,
    )


# =====================================
# Plotting FUNCTIONS
# =====================================


def plot_training_history(train_losses, val_losses, task):
    """Plot training and validation loss"""
    print(f"\n6. Plotting {task} training history...")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.title(f"{task.capitalize()} Model - Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_reg_results(pred_denorm, target_denorm):
    plt.subplot(1, 2, 1)
    plt.plot(target_denorm, label="Actual", color="blue", alpha=0.7)
    plt.plot(pred_denorm, label="Predicted", color="red", alpha=0.7)
    plt.title("Regression: Actual vs Predicted (First samples)")
    plt.xlabel("Sample")
    plt.ylabel("Signal Power [dBm]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(target_denorm, pred_denorm, alpha=0.5)
    plt.plot(
        [target_denorm.min(), target_denorm.max()],
        [target_denorm.min(), target_denorm.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Signal Power [dBm]")
    plt.ylabel("Predicted Signal Power [dBm]")
    plt.title("Regression: Scatter Plot")
    plt.grid(True, alpha=0.3)

