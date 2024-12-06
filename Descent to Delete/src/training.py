import jax.numpy as np
from jax import grad, random
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def hinge_loss(W, X, y, num_classes, l2):
    scores = np.dot(X, W)  # (num_samples, num_classes)
    correct_class_scores = scores[np.arange(len(y)), y]  # (num_samples,)
    margins = np.maximum(0, scores - correct_class_scores[:, None] + 1.0)  # Delta = 1
    margins = margins.at[np.arange(len(y)), y].set(0)  # Ignore correct class margins
    loss = np.mean(np.sum(margins, axis=1))  # Hinge loss
    reg = l2 * np.sum(W ** 2)  # Regularization term
    return loss + reg


def predict(W, X):
    scores = np.dot(X, W)
    return np.argmax(scores, axis=1)


def accuracy(W, X, y):
    y_pred = predict(W, X)
    return np.mean(y_pred == y)


def project_to_l2_ball(W, radius):
    norm = np.linalg.norm(W)
    if norm > radius:
        W = W * (radius / norm)
    return W

def step(W, X, y, num_classes, l2, learning_rate, radius):
    g = grad(hinge_loss)(W, X, y, num_classes, l2)
    W = W - learning_rate * g
    W = project_to_l2_ball(W, radius)  # Projection step
    return W

# Training loop
# def train(W, X, y, num_classes, l2=0.01, iters=100, learning_rate=0.01, radius=1.0):
def train(W, X, y, num_classes, l2, iters, learning_rate, radius):
    for _ in tqdm(range(iters), desc="Training"):
        W = step(W, X, y, num_classes, l2, learning_rate, radius)
    return W
