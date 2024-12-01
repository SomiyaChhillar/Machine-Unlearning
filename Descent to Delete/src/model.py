import jax.numpy as np
from jax import grad, random
from tqdm import tqdm

# JAX-based hinge loss
def hinge_loss(W, X, y, num_classes, l2=0.01):
    scores = np.dot(X, W)  # (num_samples, num_classes)
    correct_class_scores = scores[np.arange(len(y)), y]  # (num_samples,)
    margins = np.maximum(0, scores - correct_class_scores[:, None] + 1.0)  # Delta = 1
    margins = margins.at[np.arange(len(y)), y].set(0)  # Ignore correct class margins
    loss = np.mean(np.sum(margins, axis=1))  # Hinge loss
    reg = l2 * np.sum(W ** 2)  # Regularization term
    return loss + reg

# JAX-based prediction
def predict(W, X):
    scores = np.dot(X, W)
    return np.argmax(scores, axis=1)

# Accuracy calculation
def accuracy(W, X, y):
    y_pred = predict(W, X)
    return np.mean(y_pred == y)

# Projection to L2 ball
def project_to_l2_ball(W, radius=1.0):
    norm = np.linalg.norm(W)
    if norm > radius:
        W = W * (radius / norm)
    return W

# Single training step with projection
def step(W, X, y, num_classes, l2=0.01, learning_rate=0.01, radius=1.0):
    g = grad(hinge_loss)(W, X, y, num_classes, l2)
    W = W - learning_rate * g
    W = project_to_l2_ball(W, radius)  # Projection step
    return W

# Training loop
def train(W, X, y, num_classes, l2=0.01, iters=100, learning_rate=0.01, radius=1.0):
    for _ in tqdm(range(iters), desc="Training"):
        W = step(W, X, y, num_classes, l2, learning_rate, radius)
    return W