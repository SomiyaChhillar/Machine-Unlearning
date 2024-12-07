from tensorflow.keras.datasets import cifar10
import jax.numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_cifar10(preprocess=True, augment=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = y_train.flatten()  # Flatten labels to match expectations
    y_test = y_test.flatten()

    if augment:
            # Augmentation requires the original 4D shape
            datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True
            )
            datagen.fit(X_train)  # Keep shape as (num_samples, 32, 32, 3)
            print("Data augmentation enabled. Use datagen.flow to fetch augmented batches.")

    if preprocess:
        # Normalize and reshape
        X_train = X_train.reshape(X_train.shape[0], -1) / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1) / 255.0

        # Mean subtraction
        mean_pixel = np.mean(X_train, axis=0)
        X_train -= mean_pixel
        X_test -= mean_pixel

        # Add bias term
        bias_train = np.ones((X_train.shape[0], 1))  # Create bias column for training
        bias_test = np.ones((X_test.shape[0], 1))  # Create bias column for testing
        X_train = np.concatenate([X_train, bias_train], axis=1)
        X_test = np.concatenate([X_test, bias_test], axis=1)

        
    bias_train = np.ones((X_train.shape[0], 1))  # Bias term
    bias_test = np.ones((X_test.shape[0], 1))
    X_train = np.concatenate([X_train, bias_train], axis=1)
    X_test = np.concatenate([X_test, bias_test], axis=1)

    return (X_train, y_train), (X_test, y_test)