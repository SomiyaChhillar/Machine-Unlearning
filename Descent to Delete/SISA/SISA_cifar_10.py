import tensorflow as tf
from tensorflow.keras import datasets, layers, models, regularizers
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
import matplotlib.pyplot as plt

# 1. Data Loading and Preprocessing
def load_and_preprocess_cifar10(validation_split=0.1):
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    train_labels = train_labels.flatten()
    test_labels = test_labels.flatten()

    # Shuffle training data
    train_images, train_labels = shuffle(train_images, train_labels, random_state=42)

    # Create validation set
    num_validation_samples = int(validation_split * len(train_images))
    validation_images = train_images[:num_validation_samples]
    validation_labels = train_labels[:num_validation_samples]
    train_images = train_images[num_validation_samples:]
    train_labels = train_labels[num_validation_samples:]

    return (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels)

# 2. Partition Data into Shards and Slices
def partition_into_shards_and_slices(images, labels, num_shards, num_slices):
    shard_size = len(images) // num_shards
    shards = [
        (images[i * shard_size:(i + 1) * shard_size], labels[i * shard_size:(i + 1) * shard_size])
        for i in range(num_shards)
    ]

    # Further partition shards into slices
    all_slices = []
    for shard_images, shard_labels in shards:
        slice_size = len(shard_images) // num_slices
        slices = [
            (shard_images[i * slice_size:(i + 1) * slice_size], shard_labels[i * slice_size:(i + 1) * slice_size])
            for i in range(num_slices)
        ]
        all_slices.append(slices)
    return all_slices

# 3. Data Augmentation and Callbacks
def get_data_augmentation_and_callbacks():
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    def lr_schedule(epoch):
        lr = 1e-3
        if epoch > 80: lr *= 0.5e-3
        elif epoch > 60: lr *= 1e-3
        elif epoch > 40: lr *= 1e-2
        elif epoch > 20: lr *= 1e-1
        print(f'Epoch {epoch + 1}: Learning rate is {lr}')
        return lr

    lr_scheduler = LearningRateScheduler(lr_schedule)
    return datagen, [early_stopping, lr_scheduler]

# 4. Model Architecture
def create_cnn_model():
    weight_decay = 1e-4
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 5. Train SISA Models
def train_sisa_models(all_slices, validation_data, datagen, callbacks, epochs_per_slice=100, batch_size=64):
    shard_models = []
    histories = []
    
    for shard_idx, shard_slices in enumerate(all_slices):
        print(f"\nTraining models for Shard {shard_idx + 1}/{len(all_slices)}")
        model = create_cnn_model()
        for slice_idx, (slice_images, slice_labels) in enumerate(shard_slices):
            print(f"Training Slice {slice_idx + 1}/{len(shard_slices)} in Shard {shard_idx + 1}")
            datagen.fit(slice_images)
            history = model.fit(
                datagen.flow(slice_images, slice_labels, batch_size=batch_size),
                validation_data=validation_data,
                epochs=epochs_per_slice,
                callbacks=callbacks,
                verbose=1
            )
            histories.append(history)
        shard_models.append(model)
    return shard_models, histories

# 6. Evaluate Models
def evaluate_ensemble(models, test_images, test_labels):
    predictions = np.array([model.predict(test_images) for model in models])
    avg_predictions = np.mean(predictions, axis=0)
    ensemble_preds = np.argmax(avg_predictions, axis=1)
    ensemble_accuracy = accuracy_score(test_labels, ensemble_preds)
    print(f"\nEnsemble Accuracy: {ensemble_accuracy * 100:.2f}%")
    return ensemble_preds

# Main Function
def main():
    (train_images, train_labels), (validation_images, validation_labels), (test_images, test_labels) = load_and_preprocess_cifar10()
    all_slices = partition_into_shards_and_slices(train_images, train_labels, num_shards=2, num_slices=1)
    datagen, callbacks = get_data_augmentation_and_callbacks()
    shard_models, histories = train_sisa_models(all_slices, (validation_images, validation_labels), datagen, callbacks)
    evaluate_ensemble(shard_models, test_images, test_labels)

if __name__ == "__main__":
    main()
