# Machine Unlearning: Influence Algorithm

This project implements **Machine Unlearning** using the **Influence Algorithm** on two datasets: **CIFAR-10** and **MNIST**.

## Files

- `Influence_Algo_CIFAR10.ipynb`: Runs the Influence Algorithm on CIFAR-10 (color images).
- `Influence_Algo_MNIST.ipynb`: Applies the Influence Algorithm to MNIST (grayscale digits).

## How to Use

1. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. Run all cells to:
   - Train a model.
   - Compute influences.
   - Perform unlearning.

## Results

The notebooks show:
- Model accuracy before and after unlearning.
- Visualizations of unlearning effects.

## Requirements

- Python 3.8+
- Libraries: TensorFlow/PyTorch, NumPy, Matplotlib, Scikit-learn.
