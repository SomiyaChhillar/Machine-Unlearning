
import numpy as np 
from scipy.optimize import curve_fit
import nolds 

def state_dicts_to_1d_arrays(dict1, dict2, dict3, dict4):
    # Ensure the keys are consistent and sorted
    keys = sorted(dict1.keys())
    assert keys == sorted(dict2.keys()) == sorted(dict3.keys()) == sorted(dict4.keys()), "State dicts must have the same keys."
    
    # Initialize lists to store the 1D arrays for each state dict
    arrays_dict1, arrays_dict2, arrays_dict3, arrays_dict4 = [], [], [], []
    
    for key in keys:
        # Get the corresponding tensor values from each state dict
        tensor1, tensor2, tensor3, tensor4 = dict1[key], dict2[key], dict3[key], dict4[key]
        
        # Move tensors to CPU if necessary and convert to NumPy arrays
        tensor1 = tensor1.detach().cpu().numpy()
        tensor2 = tensor2.detach().cpu().numpy()
        tensor3 = tensor3.detach().cpu().numpy()
        tensor4 = tensor4.detach().cpu().numpy()
        
        # Flatten tensors to 1D arrays
        arrays_dict1.append(tensor1.flatten())
        arrays_dict2.append(tensor2.flatten())
        arrays_dict3.append(tensor3.flatten())
        arrays_dict4.append(tensor4.flatten())
    
    # Concatenate all 1D arrays to form a single 1D array for each state dict
    final_array1 = np.concatenate(arrays_dict1)
    final_array2 = np.concatenate(arrays_dict2)
    final_array3 = np.concatenate(arrays_dict3)
    final_array4 = np.concatenate(arrays_dict4)
    
    return final_array1, final_array2, final_array3, final_array4


def compute_temporal_influence(dict1, dict2): 
    keys = dict1.keys()
    assert keys == dict2.keys(), "State dicts must have the same keys."
    
    # Initialize lists to store the 1D arrays for each state dict
    arrays_dict1, arrays_dict2= [], []
    
    for key in keys:
        # Get the corresponding tensor values from each state dict
        tensor1, tensor2= dict1[key], dict2[key]
        
        # Move tensors to CPU if necessary and convert to NumPy arrays
        tensor1 = tensor1.cpu().numpy() if tensor1.is_cuda or tensor1.device.type == "mps" else tensor1.numpy()
        tensor2 = tensor2.cpu().numpy() if tensor2.is_cuda or tensor2.device.type == "mps" else tensor2.numpy()
        
        # Flatten tensors to 1D arrays
        arrays_dict1.append(tensor1.flatten())
        arrays_dict2.append(tensor2.flatten())
    
    # Concatenate all 1D arrays to form a single 1D array for each state dict
    final_array1 = np.concatenate(arrays_dict1)
    final_array2 = np.concatenate(arrays_dict2)
    
    difference = final_array1 - final_array2
    return np.sum(np.abs(difference))
    
    

def compute_l1_norm_difference(array1, array2, array3, array4):
    """
    Compute the L1 norm of the difference between the differences of two pairs of arrays.

    Args:
        array1 (np.ndarray): The first array.
        array2 (np.ndarray): The second array.
        array3 (np.ndarray): The third array.
        array4 (np.ndarray): The fourth array.

    Returns:
        float: The L1 norm of the final difference.
    """
    # Compute the pairwise differences
    diff1 = array1 - array2
    diff2 = array3 - array4
    
    # Compute the difference between the two differences
    final_difference = diff1 - diff2
    
    # Compute the L1 norm of the final difference
    l1_norm = np.linalg.norm(final_difference, ord=1)
    
    return l1_norm


import numpy as np
import nolds
from scipy.optimize import curve_fit
import warnings

def power_law(x, a, b, h):
    # Power-law function: Y = a * x^(-h) + b
    return a * x**(-h) + b

def compute_derivative(a, h, x):
    # Derivative of the power-law function w.r.t. x:
    # d/dx [a * x^(-h) + b] = -a * h * x^(-h-1)
    return a * (-h) * x**(-(h + 1))

def should_stop_retraining(deltas, start_block, epsilon):
    """
    Determines whether to stop retraining based on the residual memory (deltas) sequence.

    Args:
        deltas (list): Sequence of residual memory values (delta_k) after retraining each block.
        start_block (int): The starting block for retraining.
        epsilon (float): Threshold for stopping.

    Returns:
        bool: True if retraining should stop, False otherwise.
    """

    # Require more than a minimal number of points for a stable DFA calculation.
    # Previously 10 was too low; try at least 20 or more.
    if len(deltas) < 15:
        print(f"Insufficient deltas for stable DFA. Current number of deltas: {len(deltas)}")
        return False

    # Compute the DFA exponent h
    try:
        h = nolds.dfa(deltas)
    except ValueError as e:
        print(f"Error during DFA computation: {e}")
        return False

    # Ensure h is positive and not too close to zero to avoid extreme powers
    h = max(h, 0.1)

    # Choose stable initial guesses for the curve fit
    # Ensure a > 0, and set b to the last delta value
    a0 = np.mean(deltas) if np.mean(deltas) > 0 else 1.0
    b0 = deltas[-1]
    initial_guess = [a0, b0, h]

    # Create x_values starting from start_block+1 to ensure x >= 1,
    # preventing overflow from x^(-h) when x is too small
    x_values = np.arange(start_block + 1, start_block + len(deltas) + 1)

    # Add parameter bounds to keep a≥0, h≥0 and avoid overflow or negative powers that cause instability
    # Also increase maxfev to give curve_fit more chances to converge
    try:
        params, _ = curve_fit(
            power_law, x_values, deltas, p0=initial_guess,
            bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
    except RuntimeError as e:
        print(f"Error during curve fitting: {e}")
        return False

    a, b, h = params
    last_x = x_values[-1]

    # Compute the derivative at the last x value
    derivative = compute_derivative(a, h, last_x)

    # If the absolute value of the derivative is below epsilon, we consider the sequence stable
    if abs(derivative) < epsilon:
        return True

    return False
