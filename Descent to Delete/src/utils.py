import jax.numpy as np
from jax import random

# Delete a specific index from dataset
def delete_index(idx, X, y):
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    return X, y

# Append new data to the dataset
def append_datum(new_X, new_y, X, y):
    X = np.concatenate((X, new_X), axis=0)
    y = np.concatenate((y, new_y), axis=0)
    return X, y

# Process a single update (add or delete points)
def process_update(W, X, y, update, train_fn):
    X, y = update(X, y)  # Update the dataset
    print(X.shape)
    print(y.shape)
    W = train_fn(W, X, y)  # Re-train the model on the updated dataset
    return W, X, y

# Process a sequence of updates
def process_updates(W, X, y, updates, train_fn):
    for update in updates:
        W, X, y = process_update(W, X, y, update, train_fn)
    return W, X, y

# Compute sigma for publishing
def compute_sigma(num_examples, iterations, lipshitz, strong, epsilon, delta):
    gamma = (smooth - strong) / (smooth + strong)
    numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
    denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * (
        (np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta))
    )
    return numerator / denominator

# Publishing the weights with noise
def publish(rng, W, sigma):
    return W + sigma * random.normal(rng, W.shape)