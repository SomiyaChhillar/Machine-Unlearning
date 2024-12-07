import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load and preprocess data
def load_and_preprocess_data(validation_split=0.1):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    # Normalize and reshape
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # Shuffle the data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
    
    # Split validation set
    num_validation_samples = int(validation_split * len(train_images))
    validation_images = train_images[:num_validation_samples]
    validation_labels = train_labels[:num_validation_samples]
    train_images = train_images[num_validation_samples:]
    train_labels = train_labels[num_validation_samples:]
    
    return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)

# Partition data into shards and slices
def partition_data(images, labels, num_shards, num_slices):
    total_samples = len(images)
    shard_size = total_samples // num_shards
    shards = []
    
    # Create shards
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i != num_shards - 1 else total_samples
        shard_images = images[start_idx:end_idx]
        shard_labels = labels[start_idx:end_idx]
        shards.append((shard_images, shard_labels))
    
    # Create slices for each shard
    all_slices = []
    for shard_images, shard_labels in shards:
        slice_size = len(shard_images) // num_slices
        slices = [
            (shard_images[i * slice_size: (i + 1) * slice_size], 
             shard_labels[i * slice_size: (i + 1) * slice_size])
            for i in range(num_slices)
        ]
        all_slices.append(slices)
    return all_slices

# Build CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Train the SISA model
def train_sisa(all_slices, validation_data, epochs_per_slice=5, batch_size=32):
    shard_models = []
    histories = []
    
    for shard_idx, shard_slices in enumerate(all_slices):
        print(f"\nTraining Shard {shard_idx + 1}/{len(all_slices)}")
        model = create_model()
        
        for slice_idx, (slice_images, slice_labels) in enumerate(shard_slices):
            print(f"Training Slice {slice_idx + 1}/{len(shard_slices)} of Shard {shard_idx + 1}")
            history = model.fit(
                slice_images, slice_labels,
                epochs=epochs_per_slice,
                batch_size=batch_size,
                validation_data=validation_data,
                verbose=1
            )
            histories.append(history)
        shard_models.append(model)
    return shard_models, histories

# Make ensemble predictions
def ensemble_predictions(models, data):
    predictions = np.array([model.predict(data) for model in models])
    avg_predictions = np.mean(predictions, axis=0)
    return np.argmax(avg_predictions, axis=1)

# Simulate unlearning
def unlearn_label(shard_slices, label_to_remove):
    for i in range(len(shard_slices)):
        slice_images, slice_labels = shard_slices[i]
        mask = slice_labels != label_to_remove
        shard_slices[i] = (slice_images[mask], slice_labels[mask])
    return shard_slices

# Plot training results
def plot_training(histories):
    plt.figure(figsize=(14, 6))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Train {i+1}')
        plt.plot(history.history['val_accuracy'], label=f'Val {i+1}', linestyle='--')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Train {i+1}')
        plt.plot(history.history['val_loss'], label=f'Val {i+1}', linestyle='--')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Load and preprocess data
    (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = load_and_preprocess_data()
    
    # Partition data
    num_shards = 1
    num_slices = 1
    all_slices = partition_data(train_images, train_labels, num_shards, num_slices)
    
    # Train SISA
    shard_models, histories = train_sisa(all_slices, (validation_images, validation_labels))
    
    # Evaluate ensemble accuracy
    ensemble_preds = ensemble_predictions(shard_models, test_images)
    ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
    print(f"\nEnsemble Model Test Accuracy: {ensemble_accuracy * 100:.2f}%")
    
    # Simulate unlearning
    label_to_remove = 0
    unlearn_shard_idx = 0
    print(f"\nSimulating unlearning for label '{label_to_remove}' in Shard {unlearn_shard_idx + 1}")
    all_slices[unlearn_shard_idx] = unlearn_label(all_slices[unlearn_shard_idx], label_to_remove)
    
    # Retrain the affected shard
    print(f"\nRetraining Shard {unlearn_shard_idx + 1} after unlearning")
    model = create_model()
    for slice_images, slice_labels in all_slices[unlearn_shard_idx]:
        if len(slice_labels) > 0:
            model.fit(slice_images, slice_labels, epochs=5, batch_size=32, validation_data=(validation_images, validation_labels), verbose=1)
    shard_models[unlearn_shard_idx] = model
    
    # Reevaluate ensemble accuracy
    ensemble_preds = ensemble_predictions(shard_models, test_images)
    ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
    print(f"\nEnsemble Model Test Accuracy after Unlearning: {ensemble_accuracy * 100:.2f}%")
    
    plot_training(histories)

if __name__ == "__main__":
    main()
