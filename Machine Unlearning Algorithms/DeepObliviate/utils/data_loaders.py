
import os
import torch
from torchvision import datasets, transforms


def load_mnist_data_deep_obliviate(data_dir="datasets/", train_ratio=0.8):
    """
    Load and partition the MNIST dataset into training and testing tensors, resized for LeNet-5.

    Args:
        data_dir (str): Directory where the MNIST dataset will be stored.
        train_ratio (float): Ratio of data to use for training (default: 0.8).

    Returns:
        train_X (torch.Tensor): Training data features (reshaped to 1x32x32).
        train_y (torch.Tensor): Training data labels.
        test_X (torch.Tensor): Testing data features (reshaped to 1x32x32).
        test_y (torch.Tensor): Testing data labels.
    """
    

    # Ensure the directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Define transformation for MNIST (resizing and normalization)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),                 # Resize to 32x32 for LeNet-5
        transforms.ToTensor(),                      # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))        # Normalize to [-1, 1]
    ])

    # Download and load MNIST dataset
    full_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)

    # Calculate sizes for training and testing splits
    train_size = int(len(full_dataset) * train_ratio)
    test_size = len(full_dataset) - train_size

    # Split into training and testing datasets
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Convert train and test datasets to tensors
    train_X = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
    train_y = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_X = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    return train_X, train_y, test_X, test_y