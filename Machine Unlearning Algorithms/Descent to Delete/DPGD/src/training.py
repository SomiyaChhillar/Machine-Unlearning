from jax import grad
from tqdm import tqdm
import jax.numpy as jnp
from utils import project_to_l2_ball
from jax import random

def cross_entropy_loss(W, X, y, num_classes, l2):
    scores = jnp.dot(X, W)
    scores -= jnp.max(scores, axis=1, keepdims=True)  # Numerical stability
    exp_scores = jnp.exp(scores)
    probs = exp_scores / jnp.sum(exp_scores, axis=1, keepdims=True)
    # Clip probabilities to prevent log(0) and ensure stability
    # probs = jnp.clip(probs, 1e-12, 1.0)
    probs = jnp.clip(probs, 1e-12, 1.0 - 1e-12)
    correct_log_probs = -jnp.log(probs[jnp.arange(len(y)), y])
    data_loss = jnp.mean(correct_log_probs)
    reg_loss = l2 * jnp.sum(W ** 2)
    return data_loss + reg_loss

def step(W, X, y, num_classes, l2, learning_rate, radius):
    g = grad(cross_entropy_loss)(W, X, y, num_classes, l2)
    W = W - learning_rate * g
    return project_to_l2_ball(W, radius)

def train_partition(W, X, y, num_classes, l2, iters, learning_rate, radius):
    losses = []
    for _ in range(iters):
        loss = cross_entropy_loss(W, X, y, num_classes, l2)
        losses.append(loss)
        W = step(W, X, y, num_classes, l2, learning_rate, radius)
    return W, losses

def distributed_training(X, y, num_classes, num_partitions, l2, iters, learning_rate, radius, rng):
    partition_size = len(X) // num_partitions
    partitions = [(X[i * partition_size:(i + 1) * partition_size],
                   y[i * partition_size:(i + 1) * partition_size])
                  for i in range(num_partitions)]
    models = []
    all_losses = []

    with tqdm(total=num_partitions, desc="Training Partitions") as pbar:
        for idx, (X_part, y_part) in enumerate(partitions):
            # W = random.normal(rng, shape=(X.shape[1], num_classes))
            W = random.normal(rng, shape=(X_part.shape[1], num_classes)) * jnp.sqrt(2.0 / (X_part.shape[1] + num_classes))
            W, losses = train_partition(W, X_part, y_part, num_classes, l2, iters, learning_rate, radius)
            models.append(W)  # Ensure only W is appended
            all_losses.append(losses)
            pbar.update(1)

    print(f"Models after distributed training: {[type(model) for model in models]}")  # Debugging
    return models, partitions, all_losses


def reservoir_sampling_delete(S, to_delete, rng):
    if S.ndim == 1:
        # 1D case: directly replace matching elements
        random_index = random.randint(rng, shape=(), minval=0, maxval=S.shape[0])
        random_replacement = S[random_index]  # Select a random element
        S = np.where(S == to_delete, random_replacement, S)
    else:
        # 2D case: perform row-wise replacement
        random_index = random.randint(rng, shape=(), minval=0, maxval=S.shape[0])
        random_replacement = S[random_index]  # Select a random row
        # Create a boolean condition for rows matching `to_delete`
        condition = np.all(S == to_delete[None, :], axis=1)
        # Replace rows matching `to_delete`
        S = np.where(condition[:, None], random_replacement[None, :], S)
    return S


def train_affected_partitions(models, partitions, affected_indices, num_classes, l2, iters, learning_rate, radius):
    for idx in affected_indices:
        partition_X, partition_y = partitions[idx]
        W = models[idx]
        W, _ = train_partition(W, partition_X, partition_y, num_classes, l2, iters, learning_rate, radius)
        models[idx] = W  # Ensure only `W` (not a tuple) is stored
    return models
    
def publish(models, sigma, rng):
    # Log and validate shapes
    shapes = [model.shape for model in models]
    print(f"Model shapes: {shapes}")  # Debugging shapes

    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError(f"All models must have the same shape. Found shapes: {shapes}")

    # Ensure all models are JAX arrays
    models = [jnp.array(model) for model in models]

    # Stack models and calculate the average
    avg_model = jnp.mean(jnp.stack(models), axis=0)
    noise = sigma * random.normal(rng, avg_model.shape)
    # return avg_model + noise
    return avg_model

def perturbed_distributed_descent(X, y, num_classes, l2, iters, learning_rate, radius, num_partitions, sigma, rng, delete_indices):
    # Initial distributed training
    models, partitions, all_losses = distributed_training(X, y, num_classes, num_partitions, l2, 1500, learning_rate, radius, rng)

    # Debugging: Log types and shapes of models before affected partitions update
    print(f"Models before affected partitions update: {[type(model) for model in models]}")

    # Identify affected partitions
    partition_size = len(X) // num_partitions
    affected_partitions = set(idx // partition_size for idx in delete_indices)

    # Update only the affected partitions
    models = train_affected_partitions(models, partitions, affected_partitions, num_classes, l2, iters, learning_rate, radius)

    # Debugging: Log types and shapes of models after affected partitions update
    print(f"Models after affected partitions update: {[type(model) for model in models]}")

    # Publish the perturbed model
    W_published = publish(models, sigma, rng)
    return W_published, all_losses