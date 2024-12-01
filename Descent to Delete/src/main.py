import jax.numpy as np
from jax import random
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import train, predict, accuracy
from utils import delete_index, process_updates, compute_sigma, publish

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess MNIST
    X_train = X_train.reshape(X_train.shape[0], -1) / 255.0  # Normalize and flatten
    X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # JAX-based SVM training
    num_classes = 10
    num_features = X_train.shape[1]
    rng = random.PRNGKey(42)
    W = random.normal(rng, shape=(num_features, num_classes))

    # Hyperparameters
    l2 = 0.01
    learning_rate = 0.01
    iters = 1000
    radius = 1.0

    # Parameters for differential privacy
    num_examples = X_train.shape[0]
    lipshitz = 1 + l2
    strong = l2
    smooth = 4 - l2
    epsilon = 5
    delta = 1 / (num_examples ** 2)

    # Compute sigma for publishing
    sigma = compute_sigma(num_examples, iters, lipshitz, strong, epsilon, delta)

    # Train the model
    W = train(W, X_train, y_train, num_classes, l2, 50, learning_rate, radius)
     # Configurable indices to delete

    # Training function
    train_fn = lambda W, X, y: train(W, X, y, num_classes, l2, 10, learning_rate, radius)

    indices_to_delete = [0, 1, 5, 10]  # Indices you want to delete from the dataset

    # Create updates for deleting indices
    updates = [lambda X, y, idx=idx: delete_index(idx, X, y) for idx in indices_to_delete]

    # Process updates and retrain
    W, X_train, y_train = process_updates(W, X_train, y_train, updates, train_fn)

    W_retrained = random.normal(rng, shape=(num_features, num_classes))
    W_retrained = train(W_retrained, X_train, y_train, num_classes, l2, 50, learning_rate, radius)


    # Add noise to weights for publishing
    temp_rng, rng = random.split(rng)
    W_published = publish(temp_rng, W, sigma)

    # Final evaluation on test set
    test_acc = accuracy(W_published, X_test, y_test)
    print(f"Test Accuracy (Published Weights): {test_acc:.4f}")

    # Confusion matrix for published weights
    y_pred_published = predict(W_published, X_test)
    conf_matrix = confusion_matrix(y_test, y_pred_published)
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Published Weights)')
    plt.show()