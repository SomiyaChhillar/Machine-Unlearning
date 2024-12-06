import jax.numpy as np
from jax import random

def project_to_l2_ball(W, radius):
    norm = np.linalg.norm(W)
    if norm > radius:
        W = W * (radius / norm)
    return W


def predict(W, X):
    scores = np.dot(X, W)
    return np.argmax(scores, axis=1)


def evaluate_accuracy(W, X, y):
    y_pred = predict(W, X)
    accuracy = np.mean(y_pred == y)
    return accuracy