import jax.numpy as np
from jax import random
from jax.numpy.linalg import norm

def delete_index(idx, X, y):
    X = np.delete(X, idx, axis=0)
    y = np.delete(y, idx, axis=0)
    return X, y

def append_datum(new_X, new_y, X, y):
    X = np.concatenate((X, new_X), axis=0)
    y = np.concatenate((y, new_y), axis=0)
    return X, y

def process_update(W, X, y, update, train_fn):
    X, y = update(X, y)  # Update the dataset
    print(X.shape)
    print(y.shape)
    W= train_fn(W, X, y)  # Re-train the model on the updated dataset
    return W, X, y


def process_updates(W, X, y, updates, train_fn):    
    for update in updates:
        W, X, y = process_update(W, X, y, update, train_fn)
    return W, X, y


def compute_sigma(num_examples, iterations, lipshitz, smooth, strong, epsilon, delta):
    gamma = (smooth - strong) / (smooth + strong)
    numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
    denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * (
        (np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta))
    )
    return numerator / denominator


def publish(rng, W, sigma):
    return W + sigma * random.normal(rng, W.shape)

def compute_constants(X, l2, epsilon, delta, d, n, I):
    max_norm = np.max(np.linalg.norm(X, axis=1))
    lipschitz = max_norm / X.shape[0] + l2
    spectral_norm = norm(X, ord=2) ** 2
    smooth = spectral_norm / X.shape[0] + l2

    strong = l2
    return lipschitz, smooth, strong


