from load_data import load_cifar10
from training import perturbed_distributed_descent
from utils import evaluate_accuracy
from jax import random
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Load CIFAR-10 with preprocessing
    (X_train, y_train), (X_test, y_test) = load_cifar10()
    
    # Parameters
    num_samples, num_features, num_classes = X_train.shape[0], X_train.shape[1], 10
    rng = random.PRNGKey(42)

    # Hyperparameters
    l2 = 0.001
    iters = 100
    learning_rate = 0.01
    radius = 5.0
    num_partitions = 10
    sigma = 0.1

    # Indices to delete
    delete_indices = [0, 100, 500]  # Replace with desired indices for deletion

    # Train with perturbed distributed descent
    W_published, all_losses = perturbed_distributed_descent(
        X_train, y_train, num_classes, l2, iters, learning_rate, radius, num_partitions, sigma, rng, delete_indices
    )
