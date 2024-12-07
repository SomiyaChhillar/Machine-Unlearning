from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import numpy as np
from collections import defaultdict
import torch 
import os
import json 
from utils.general_utils import generate_unique_folder_name
from utils.plotting_utils import plot_losses 
from utils.deepObliviateUtils import * 
import shutil
import warnings 
import hashlib
import torch.optim as optim

class DeepObliviate:
    def __init__(self, model, params, device=None):
        """
        Initialize the DeepObliviate class.

        Args:
            model (torch.nn.Module): The PyTorch model to train and unlearn.
            params (dict): A dictionary of training parameters.
            device (torch.device): The device to use for training (default: automatically detected).
        """
        # Select the device for computation: use MPS (Apple Silicon GPU) if available, else CPU
        self.device = device or (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

        # Move the model to the selected device
        self.model = model.to(self.device)
        self.params = params

        # Extract training parameters
        self.num_blocks = params['num_blocks']          # Number of blocks to partition the dataset into
        self.epochs_per_block = params['epochs_per_block']  # Number of epochs to train on each block
        self.optimizer = params['optimizer']            # Optimizer for training
        self.criterion = params['criterion']            # Loss function

        # Initialize mapping from data indices to block indices
        self.index_to_block = {}

        # Initialize list to store dataset blocks
        self.blocks = []

    def save_optimizer_state(self, block_idx, folder_path):
        """
        Save the optimizer state to a file.

        Args:
            block_idx (int): The index of the block.
            folder_path (str): The folder where the optimizer state will be saved.
        """
        # Construct the file path for saving the optimizer state
        optimizer_path = os.path.join(folder_path, f"optimizer_block_{block_idx}.pth")
        # Save the optimizer state dictionary to the specified file
        torch.save(self.optimizer.state_dict(), optimizer_path)

    def load_optimizer_state(self, block_idx, folder_path):
        """
        Load the optimizer state from a file.

        Args:
            block_idx (int): The index of the block.
            folder_path (str): The folder where the optimizer state is stored.
        """
        # Construct the file path for loading the optimizer state
        optimizer_path = os.path.join(folder_path, f"optimizer_block_{block_idx}.pth")
        if os.path.exists(optimizer_path):
            # Load the optimizer state dictionary from the file
            self.optimizer.load_state_dict(torch.load(optimizer_path))
            print(f"Loaded optimizer state for Block {block_idx}.")
                
    def partition_into_blocks(self, X, y):
        """
        Partition the dataset into blocks while preserving original indices.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            list: A list of DataLoader objects, one for each block.
        """
        # Create a dictionary to store indices of samples for each class
        class_indices = defaultdict(list)
        for idx, label in enumerate(y):
            # Map each class label to the list of its sample indices
            class_indices[label.item()].append(idx)

        # Shuffle indices within each class to ensure randomness
        for label in class_indices:
            np.random.shuffle(class_indices[label])

        # Initialize a list to store indices for each block
        block_indices = [[] for _ in range(self.num_blocks)]
        for label, indices in class_indices.items():
            # Calculate the base size of each block for the current class
            block_sizes = [len(indices) // self.num_blocks] * self.num_blocks
            remainder = len(indices) % self.num_blocks

            # Distribute the remainder among the first few blocks
            for i in range(remainder):
                block_sizes[i] += 1

            # Assign indices to blocks
            start = 0
            for block_idx, size in enumerate(block_sizes):
                # Get the indices for the current block
                assigned_indices = indices[start:start + size]
                # Add the indices to the corresponding block
                block_indices[block_idx].extend(assigned_indices)

                # Update the index-to-block mapping
                for idx in assigned_indices:
                    self.index_to_block[idx] = block_idx

                start += size

        # Create Subset objects for each block with original indices
        self.blocks = [Subset(TensorDataset(X, y), indices) for indices in block_indices]
        
        print(f"BLOCK SIZES {len(self.blocks[0])}")

        # Create DataLoaders for each block
        self.data_loaders = [DataLoader(block, batch_size=self.params['batch_size'], shuffle=False) for block in self.blocks]
        return self.data_loaders

    def get_block_for_index(self, data_index):
        """
        Retrieve the block index for a specific data point.

        Args:
            data_index (int): Index of the data point in the original dataset.

        Returns:
            int: The block index the data point belongs to.
        """
        # Check if the data index exists in the mapping
        if data_index not in self.index_to_block:
            raise ValueError(f"Data index {data_index} is not assigned to any block.")
        # Return the block index for the given data index
        return self.index_to_block[data_index]

    def train(self, X, y):
        """
        Train the model on the dataset divided into blocks, track losses, and plot them.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.
        """
        # Move data to the selected device
        X = X.to(self.device)
        y = y.to(self.device)

        # Partition data into blocks and get DataLoaders
        data_loaders = self.partition_into_blocks(X, y)
        self.original_blocks = self.blocks

        # Generate a unique folder name for saving model checkpoints
        self.folder_name = generate_unique_folder_name(self.model.__class__, self.params)
        self.folder_name = os.path.join('models/unlearning/checkpoints', self.folder_name)
        full_folder_path = os.path.join(self.folder_name, 'original')
        print(f"Saving training results in folder: {full_folder_path}")
        os.makedirs(full_folder_path, exist_ok=True)
        self.checkpoint_dir = full_folder_path

        train_losses = []

        # Check for existing checkpoints to possibly resume training
        existing_checkpoints = [
            fname for fname in os.listdir(full_folder_path)
            if fname.startswith("block_") and fname.endswith(".pth")
        ]
        if existing_checkpoints:
            # Find the latest block number from existing checkpoints
            latest_block = max(
                int(fname.split('_')[1].split('.')[0]) for fname in existing_checkpoints
            )
            print(f"Resuming training from Block {latest_block + 1}.")
            # Load the model state from the latest checkpoint
            latest_checkpoint_path = os.path.join(full_folder_path, f"block_{latest_block}.pth")
            self.model.load_state_dict(torch.load(latest_checkpoint_path))
            # Load the optimizer state
            self.load_optimizer_state(latest_block, full_folder_path)
            start_block = latest_block + 1
        else:
            print("No existing checkpoints found. Starting training from Block 1.")
            start_block = 1

        # Iterate over the blocks starting from the start_block
        for block_idx in range(start_block - 1, len(data_loaders)):
            print(f"Training on Block {block_idx + 1}/{self.num_blocks}")
            self.model.train()

            # Train for the specified number of epochs per block
            for epoch in range(self.epochs_per_block):
                epoch_loss = 0.0
                # Iterate over the DataLoader for the current block
                for data, target in data_loaders[block_idx]:
                    # Move data and target to the device
                    data, target = data.to(self.device), target.to(self.device)
                    # Zero the parameter gradients
                    self.optimizer.zero_grad()
                    # Forward pass
                    outputs = self.model(data)
                    # Compute the loss
                    loss = self.criterion(outputs, target)
                    # Backward pass
                    loss.backward()
                    # Optimize
                    self.optimizer.step()
                    # Accumulate loss
                    epoch_loss += loss.item()

                # Compute average loss for the epoch
                avg_epoch_loss = epoch_loss / len(data_loaders[block_idx])
                train_losses.append(avg_epoch_loss)
                print(f"Block {block_idx + 1}, Epoch {epoch + 1}/{self.epochs_per_block}, Loss: {avg_epoch_loss:.4f}")

            # Save the model and optimizer state after training on the block
            block_filename = os.path.join(full_folder_path, f"block_{block_idx}.pth")
            torch.save(self.model.state_dict(), block_filename)
            self.save_optimizer_state(block_idx , full_folder_path)
            print(f"Saved model and optimizer state after Block {block_idx}.")

        # Plot the training losses
        plot_losses(train_losses, None, None)
        
    def get_block_data(self, block_idx):
        """
        Retrieve the data and targets for a specific block.

        Args:
            block_idx (int): The index of the block to retrieve.

        Returns:
            list: A list of dictionaries containing 'data', 'target', and 'index'.
        """
        # Validate the block index
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(f"Invalid block index: {block_idx}. Block index must be between 0 and {len(self.blocks) - 1}.")

        block = self.blocks[block_idx]
        # Extract data and targets from the block's dataset
        data, targets = block.dataset.tensors  # Extract X, y tensors
        # Get the original dataset indices for this block
        indices = block.indices

        # Create a list of dictionaries with data, target, and index
        block_data = [{"data": data[i], "target": targets[i].item(), "index": indices[i]} for i in range(len(indices))]
        return block_data

    def update_block_data(self, block_idx, new_data):
        """
        Update the dataset and DataLoader for a specific block.

        Args:
            block_idx (int): The index of the block to update.
            new_data (list): A list of dictionaries containing 'data', 'target', and 'index'.
        """
        # Validate the block index
        if block_idx < 0 or block_idx >= len(self.blocks):
            raise ValueError(f"Invalid block index: {block_idx}. Block index must be between 0 and {len(self.blocks) - 1}.")

        # Extract data and targets from new_data
        data = torch.stack([item["data"] for item in new_data])
        targets = torch.tensor([item["target"] for item in new_data], dtype=torch.long)

        # Create a new TensorDataset for the block
        new_dataset = TensorDataset(data, targets)

        # The new indices are now 0 to len(new_data)-1, since it's a new dataset
        new_indices = list(range(len(new_data)))

        # Update the block with the new dataset and indices
        self.blocks[block_idx] = Subset(new_dataset, new_indices)

        # Update the corresponding DataLoader
        self.data_loaders[block_idx] = DataLoader(
            self.blocks[block_idx],
            batch_size=self.params['batch_size'],
            shuffle=False
        )

    def retrain_block(self, block_idx):
        """
        Retrain the model on a specific block using self.data_loaders.

        Args:
            block_idx (int): The index of the block to retrain on.
        """
        # Validate the block index
        if block_idx < 0 or block_idx >= len(self.data_loaders):
            raise ValueError(f"Invalid block index: {block_idx}. Block index must be between 0 and {len(self.data_loaders) - 1}.")

        # Get the DataLoader for the specified block
        block_loader = self.data_loaders[block_idx]
        self.model.train()

        # Train for the specified number of epochs per block
        for epoch in range(self.epochs_per_block):
            epoch_loss = 0.0
            # Iterate over the DataLoader for the block
            for data, target in block_loader:
                # Move data and target to the device
                data, target = data.to(self.device), target.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(data)
                # Compute the loss
                loss = self.criterion(outputs, target)
                # Backward pass
                loss.backward()
                # Optimize
                self.optimizer.step()
                # Accumulate loss
                epoch_loss += loss.item()

            # Compute average loss for the epoch
            avg_epoch_loss = epoch_loss / len(block_loader)
            print(f"Retrain Block {block_idx}, Epoch {epoch + 1}/{self.epochs_per_block}, Loss: {avg_epoch_loss:.4f}")

    def unlearn(self, indices_to_unlearn, epsilon=1e-5):
        """
        Unlearn multiple data points by retraining from the earliest affected block onward.
        Ensures all affected blocks are retrained before applying the stopping criterion.
        Implements model stitching to avoid retraining unaffected blocks.

        Args:
            indices_to_unlearn (list): List of indices to unlearn.
            epsilon (float): Threshold for the stopping criterion.

        Returns:
            deltas (list): List of delta values for each retrained block.
        """
        # Map indices to blocks
        # Create a set of all blocks that contain data points to unlearn
        self.blocks = self.original_blocks 
        blocks_to_unlearn = set(self.get_block_for_index(idx) for idx in indices_to_unlearn)
        # Determine the earliest and latest blocks that contain data points to unlearn
        earliest_block = min(blocks_to_unlearn)
        latest_block = max(blocks_to_unlearn)

        print(f"Data points to unlearn are in blocks: {blocks_to_unlearn}")

        unique_hash = hashlib.md5(json.dumps(indices_to_unlearn).encode()).hexdigest()
        unlearn_folder = os.path.join(
            self.folder_name,
            f'unlearn_{unique_hash}')
        unlearn_folder = unlearn_folder + f'epsilon_{epsilon}'
        os.makedirs(unlearn_folder, exist_ok=True)

        deltas = []

        # Copy unaffected checkpoints and optimizer states up to earliest_block - 1
        for block in range(earliest_block):
            checkpoint_src = os.path.join(self.checkpoint_dir, f"block_{block}.pth")
            checkpoint_dest = os.path.join(unlearn_folder, f"block_{block}.pth")
            optimizer_src = os.path.join(self.checkpoint_dir, f"optimizer_block_{block}.pth")
            optimizer_dest = os.path.join(unlearn_folder, f"optimizer_block_{block}.pth")
            # Copy model checkpoint if it exists
            if os.path.exists(checkpoint_src):
                shutil.copyfile(checkpoint_src, checkpoint_dest)
            # Copy optimizer state if it exists
            if os.path.exists(optimizer_src):
                shutil.copyfile(optimizer_src, optimizer_dest)
            # Deltas for blocks before retraining are zero
            deltas.append(0)

        # Load model and optimizer state from the last unaffected block
        if earliest_block > 0:
            # Load checkpoint
            prev_checkpoint = os.path.join(self.checkpoint_dir, f"block_{earliest_block}.pth")
            self.model.load_state_dict(torch.load(prev_checkpoint))
            try:
                self.load_optimizer_state(earliest_block, self.checkpoint_dir)
            except ValueError:
                # If optimizer state doesn't match, reinitialize the optimizer
                print("Optimizer state doesn't match, reinitializing optimizer...")
                self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        else:
            # Reinitialize model and optimizer if starting from scratch
            self.model.apply(self.model._initialize_weights)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Remove data points from their respective blocks
        for block_idx in blocks_to_unlearn:
            block_data = self.get_block_data(block_idx)
            # Filter out the data points to unlearn
            filtered_data = [x for x in block_data if x["index"] not in indices_to_unlearn]
            self.update_block_data(block_idx, filtered_data)
            print(f"Removed data points from Block {block_idx}.")

        # Retrain from earliest_block to latest_block without applying stopping criterion
        for block in range(earliest_block, latest_block):
            # Retrain the model on the current block
            self.retrain_block(block)

            # Save retrained model and optimizer states
            checkpoint_path = os.path.join(unlearn_folder, f"block_{block}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            self.save_optimizer_state(block, unlearn_folder)
            print(f"Saved retrained model and optimizer state for Block {block}.")

            if block != 0: 
                # Compute delta_k (residual memory) between original and retrained models
                original_checkpoint_k = os.path.join(self.checkpoint_dir, f"block_{block}.pth")
                original_checkpoint_k_minus_1 = os.path.join(self.checkpoint_dir, f"block_{block-1}.pth")
                unlearn_checkpoint_k = os.path.join(unlearn_folder, f"block_{block}.pth")
                unlearn_checkpoint_k_minus_1 = os.path.join(unlearn_folder, f"block_{block-1}.pth")

                # Load state_dicts for original and retrained models
                state_dict_k = torch.load(original_checkpoint_k)
                state_dict_k_minus_1 = torch.load(original_checkpoint_k_minus_1)
                state_dict_k_prime = torch.load(unlearn_checkpoint_k)
                state_dict_k_minus_1_prime = torch.load(unlearn_checkpoint_k_minus_1)

                # Compute delta_k using utility functions (flattened parameter differences)
                P_k, P_k_minus1, P_k_prime, P_k_minus1_prime = state_dicts_to_1d_arrays(
                    state_dict_k, state_dict_k_minus_1, state_dict_k_prime, state_dict_k_minus_1_prime
                )
                delta_k = compute_l1_norm_difference(P_k, P_k_minus1, P_k_prime, P_k_minus1_prime)
                deltas.append(delta_k)

                print(f"Block {block + 1}: Residual Memory (Delta_k) = {delta_k:.6f}")

            # Do not apply stopping criterion during this phase

        # Continue retraining beyond latest_block, applying stopping criterion
        for block in range(latest_block, self.num_blocks):
            # Retrain the model on the current block
            self.retrain_block(block)

            # Save retrained model and optimizer states
            checkpoint_path = os.path.join(unlearn_folder, f"block_{block}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            self.save_optimizer_state(block, unlearn_folder)
            print(f"Saved retrained model and optimizer state for Block {block}.")

            if block !=0: 
                # Compute delta_k (residual memory) between original and retrained models
                original_checkpoint_k = os.path.join(self.checkpoint_dir, f"block_{block}.pth")
                original_checkpoint_k_minus_1 = os.path.join(self.checkpoint_dir, f"block_{block-1}.pth")
                unlearn_checkpoint_k = os.path.join(unlearn_folder, f"block_{block}.pth")
                unlearn_checkpoint_k_minus_1 = os.path.join(unlearn_folder, f"block_{block-1}.pth")

                # Load state_dicts for original and retrained models
                state_dict_k = torch.load(original_checkpoint_k)
                state_dict_k_minus_1 = torch.load(original_checkpoint_k_minus_1)
                state_dict_k_prime = torch.load(unlearn_checkpoint_k)
                state_dict_k_minus_1_prime = torch.load(unlearn_checkpoint_k_minus_1)

                # Compute delta_k using utility functions (flattened parameter differences)
                P_k, P_k_minus1, P_k_prime, P_k_minus1_prime = state_dicts_to_1d_arrays(
                    state_dict_k, state_dict_k_minus_1, state_dict_k_prime, state_dict_k_minus_1_prime
                )
                delta_k = compute_l1_norm_difference(P_k, P_k_minus1, P_k_prime, P_k_minus1_prime)
                deltas.append(delta_k)

                print(f"Block {block + 1}: Residual Memory (Delta_k) = {delta_k:.6f}")

            # Apply the stopping criterion after latest affected block
            if block < self.num_blocks - 1:
                stop = should_stop_retraining(
                    deltas=(deltas[earliest_block:] if earliest_block > 0 else deltas),
                    start_block=earliest_block,
                    epsilon=epsilon
                )
                if stop:
                    print(f"Stopping retraining at Block {block + 1}.")
                    break

        # Model Stitching: Combine retrained model with remaining blocks' influence
        # Load M_B (fully trained original model after all blocks)
        MB_checkpoint = os.path.join(self.checkpoint_dir, f"block_{self.num_blocks-1}.pth")
        MB_state_dict = torch.load(MB_checkpoint)

        # Load M_{d+t} (original model after retraining stops)
        Md_t_checkpoint = os.path.join(self.checkpoint_dir, f"block_{block}.pth")
        Md_t_state_dict = torch.load(Md_t_checkpoint)

        # Compute the parameter difference Delta = M_B - M_{d+t}
        Delta_state_dict = self.compute_model_difference(MB_state_dict, Md_t_state_dict)

        # Add Delta to the current retrained model M' to stitch the models
        M_prime_state_dict = self.model.state_dict()
        stitched_state_dict = self.add_difference_to_model(M_prime_state_dict, Delta_state_dict)

        # Update the model with the stitched state_dict
        self.model.load_state_dict(stitched_state_dict)

        # Save the final unlearned model after stitching
        final_model_path = os.path.join(unlearn_folder, f"final_unlearned_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Saved final unlearned model after stitching at {final_model_path}")

        return deltas


    def compute_model_difference(self, model_state_dict1, model_state_dict2):
        """
        Compute the difference between two model state dictionaries.

        Args:
            model_state_dict1 (dict): The first model's state_dict.
            model_state_dict2 (dict): The second model's state_dict.

        Returns:
            dict: A state_dict representing the difference (model_state_dict1 - model_state_dict2).
        """
        difference = {}
        for key in model_state_dict1.keys():
            # Ensure tensors are on the same device
            tensor1 = model_state_dict1[key].to(self.device)
            tensor2 = model_state_dict2[key].to(self.device)
            # Compute the difference for each parameter
            difference[key] = tensor1 - tensor2
        return difference

    def add_difference_to_model(self, model_state_dict, difference):
        """
        Add a parameter difference to a model's state_dict.

        Args:
            model_state_dict (dict): The model's state_dict to be updated.
            difference (dict): The parameter difference to add.

        Returns:
            dict: The updated model state_dict.
        """
        updated_state_dict = {}
        for key in model_state_dict.keys():
            # Ensure tensors are on the same device
            tensor = model_state_dict[key].to(self.device)
            delta = difference[key].to(self.device)
            # Add the difference to each parameter
            updated_state_dict[key] = tensor + delta
        return updated_state_dict



