# SISA Training and Unlearning on MNIST

This project demonstrates **SISA (Sharded, Isolated, Sliced, and Aggregated) training** on the MNIST dataset. It involves partitioning the dataset into shards and slices, training models incrementally, and simulating **machine unlearning** by removing specific labels and retraining affected data partitions. 

## Key Features
1. **Data Partitioning**:
   - Training data is divided into multiple **shards**, each representing an independent subset of the dataset.
   - Each shard is further partitioned into **slices** for incremental training.

2. **Model Training**:
   - A CNN model is trained on slices within each shard.
   - Validation is performed using a held-out set to monitor performance.

3. **Ensemble Predictions**:
   - Predictions are averaged across all shard models, forming an ensemble for final evaluation.
   - Ensemble accuracy is calculated on the test set.

4. **Simulated Unlearning**:
   - Demonstrates unlearning by removing all training samples of a specific label (`0` in this case) from a shard.
   - Affected slices are retrained without the removed data.
   - Post-unlearning ensemble accuracy is recalculated to assess the effect.

5. **Evaluation**:
   - Training and validation accuracy/loss plots for each slice are generated.
   - Accuracy on the removed class is checked to verify unlearning effectiveness.

## Requirements
- Python 3.7+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib

## How to Run
1. Load the script and ensure dependencies are installed.
2. Execute the script to:
   - Train the SISA model.
   - Simulate unlearning and retrain the affected shard.
   - Generate training/validation plots and evaluate ensemble accuracy.

3. Modify parameters such as the number of shards, slices, epochs, and batch size to test different configurations.
4. For detailed code understanding please have a look at notebook file.