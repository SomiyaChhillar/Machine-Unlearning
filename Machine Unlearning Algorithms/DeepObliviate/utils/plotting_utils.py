import matplotlib.pyplot as plt 
import os 

def plot_losses(train_losses, test_losses, folder_path):
    """
    Plot and save the train and test losses.

    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of testing losses.
        folder_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    if test_losses:
        plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Losses")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    if folder_path: 
        plt_path = os.path.join(folder_path, "loss_plot.png")
        plt.savefig(plt_path)
        print(f"Saved loss plot to {plt_path}")
    plt.show()
    plt.close()
   