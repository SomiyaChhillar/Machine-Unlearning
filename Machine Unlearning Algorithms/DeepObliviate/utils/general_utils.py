import hashlib
import json
from torch.utils.data import random_split
import torch 

def generate_unique_folder_name(instance_class, hyperparameters):
    """
    Generate a unique, stable folder name based on the class name and hyperparameters.

    Args:
        instance_class (class): The simulation class used.
        hyperparameters (dict): The hyperparameters used to configure the simulation.

    Returns:
        str: A unique folder name.
    """
    # Custom serialization: Convert non-serializable objects to strings
    def serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj  # Return directly serializable types
        return str(obj)  # Convert other objects to their string representation

    # Apply serialization to the hyperparameters dictionary
    serializable_hyperparameters = {k: serialize(v) for k, v in hyperparameters.items()}

    # Convert the hyperparameters dictionary to a JSON string
    hyperparams_str = json.dumps(serializable_hyperparameters, sort_keys=True)

    # Generate a unique hash
    unique_hash = hashlib.md5(hyperparams_str.encode()).hexdigest()

    # Return the folder name
    return f"{instance_class.__name__}_{unique_hash}"



def partition_dataset(train_X, train_y, train_ratio=0.8):
    """
    Partition the dataset into training and testing subsets.

    Args:
        train_X (torch.Tensor): Input data.
        train_y (torch.Tensor): Target labels.
        train_ratio (float): Ratio of data to use for training (default: 0.8).

    Returns:
        train_dataset (TensorDataset): Training dataset.
        test_dataset (TensorDataset): Testing dataset.
    """
    # Combine inputs and labels into a single TensorDataset
    dataset = torch.utils.data.TensorDataset(train_X, train_y)

    # Calculate sizes for training and testing splits
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    # Split the dataset
    train_subset, test_subset = random_split(dataset, [train_size, test_size])

    return train_subset, test_subset


def evaluate_accuracy_of_a_nn_model(model, test_X, test_y, device, batch_size=64):
    model.eval()
    correct = 0
    total = 0
    model.to(device)  # Ensure the model is on the correct device
    with torch.no_grad():
        for i in range(0, len(test_X), batch_size):
            batch_X = test_X[i:i+batch_size].to(device)  # Move batch to correct device
            batch_y = test_y[i:i+batch_size].to(device)  # Move labels to correct device
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    return correct / total