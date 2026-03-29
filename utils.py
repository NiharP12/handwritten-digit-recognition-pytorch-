import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MNISTCSVDataset(Dataset):
    """
    Custom PyTorch Dataset to load MNIST from your local CSV files.
    """
    def __init__(self, csv_file):
        print(f"[*] Loading dataset from {csv_file}... this might take a moment.")
        # Load CSV into a Pandas DataFrame
        self.data = pd.read_csv(csv_file)
        
        # Split labels (first column) and pixel values (remaining 784 columns)
        self.labels = self.data.iloc[:, 0].values
        self.pixels = self.data.iloc[:, 1:].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        pixels = self.pixels[idx]
        
        # Pixels range from 0-255, we convert them to [0, 1] mapped as a Tensor
        # We explicitly view it as 1 channel, 28x28
        img_tensor = torch.tensor(pixels / 255.0).view(1, 28, 28)
        
        # Standard MNIST Normalization calculation (Mean and Std Dev)
        img_tensor = (img_tensor - 0.1307) / 0.3081
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def get_data_loaders(batch_size=64):
    """
    Prepares PyTorch DataLoaders exclusively using local CSV files.
    No automatic PyTorch internet downloading is used!
    """
    train_dataset = MNISTCSVDataset('mnist_train.csv')
    test_dataset = MNISTCSVDataset('mnist_test.csv')

    # Create DataLoaders to implicitly batch and shuffle the CSV data
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, test_loader

def get_device():
    """
    Automatically detects if a CUDA-compatible GPU is available.
    Returns the appropriate PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
