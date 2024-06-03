import numpy as np
import matplotlib.pyplot as plt
import os

def load_x_data(x_file):
    """
    Load X-axis data from a specified file.

    Parameters:
    x_file (str): Path to the file containing X-axis data.

    Returns:
    np.ndarray: Array of X-axis values.
    """
    with open(x_file, 'r') as file:
        lines = file.readlines()
    
    # Find the end of the header
    header_end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == '#end of header':
            header_end_idx = i
            break
    
    # Read the data after the header
    x_values = [float(line.strip()) for line in lines[header_end_idx + 2:] if line.strip()]
    return np.array(x_values)

def load_xyz_data_from_directory(directory):
    """
    Load Y-axis and phase data from all .txt files in a specified directory, sorted alphabetically.

    Parameters:
    directory (str): Path to the directory containing the .txt files.

    Returns:
    np.ndarray, np.ndarray: Arrays of Y-axis (frequency) values and phase values.
    """
    y_values = []
    phase_values = []

    # Get all .txt files in the directory and sort them alphabetically
    filenames = sorted([f for f in os.listdir(directory) if f.endswith('.txt')])

    for filename in filenames:
        y_file = os.path.join(directory, filename)
        with open(y_file, 'r') as file:
            lines = file.readlines()

        # Find the end of the header
        header_end_idx = 0
        for i, line in enumerate(lines):
            if line.strip() == '#end of header':
                header_end_idx = i
                break

        # Read the data after the header
        data = [line.strip().split() for line in lines[header_end_idx + 2:] if line.strip()]
        data = np.array(data, dtype=float)

        if data.shape[1] >= 3:  # Ensure there are at least three columns
            y_values.append(data[:, 0])     # First column: frequency
            phase_values.append(data[:, 2]) # Third column: phase

    y_values = np.array(y_values)
    phase_values = np.array(phase_values)
    return y_values, phase_values

def load_data(base_name):
    """
    Load X-axis, Y-axis, and phase data based on a base name. 

    Parameters:
    base_name (str): Base name of the files and directory.

    Returns:
    np.ndarray, np.ndarray, np.ndarray: Arrays of X-axis values, Y-axis values, and phase values.
    """
    # Define paths to the files and directories based on the base name
    x_file = f'{base_name}.txt'
    y_directory = f'{base_name}-children'

    # Load X-axis data
    x_values = load_x_data(x_file)
    # Load Y-axis and phase data from the directory
    y_values, phase_values = load_xyz_data_from_directory(y_directory)

    return x_values, y_values, phase_values

def remove_column_offsets(matrix):
    """
    Remove the mean of each column from the matrix to eliminate column offsets.

    Parameters:
    matrix (np.ndarray): Input matrix.

    Returns:
    np.ndarray: Adjusted matrix with column offsets removed.
    """
    column_means = np.mean(matrix, axis=0)
    adjusted_matrix = matrix - column_means
    return adjusted_matrix

def remove_row_offsets(matrix):
    """
    Remove the mean of each row from the matrix to eliminate row offsets.

    Parameters:
    matrix (np.ndarray): Input matrix.

    Returns:
    np.ndarray: Adjusted matrix with row offsets removed.
    """
    row_means = np.mean(matrix, axis=1, keepdims=True)
    adjusted_matrix = matrix - row_means
    return adjusted_matrix

# def plot_phase_map(x_values, y_values, phase_matrix, title='Phase Map', xlabel='X-axis', ylabel='Frequency Index (Y)'):
#     """
#     Plot a 2D phase map using the provided X-axis values, Y-axis values, and phase matrix.

#     Parameters:
#     x_values (np.ndarray): Array of X-axis values.
#     y_values (np.ndarray): Array of Y-axis values.
#     phase_matrix (np.ndarray): Matrix of phase values.
#     title (str): Title of the plot.
#     xlabel (str): Label for the X-axis.
#     ylabel (str): Label for the Y-axis.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.imshow(phase_matrix, aspect='auto',extent=[x_values[0], x_values[-1], 0, len(y_values)], origin='lower', cmap='viridis')
#     plt.colorbar(label='Phase')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.show()

def plot_phase_map(x_values, y_values_list, phase_matrix, title='Phase Map', xlabel='X-axis', ylabel='Frequency (Y)'):
    """
    Plot a 2D phase map using the provided X-axis values, Y-axis values, and phase matrix.

    Parameters:
    x_values (np.ndarray): Array of X-axis values.
    y_values_list (list): List of Y-axis arrays.
    phase_matrix (np.ndarray): Matrix of phase values.
    title (str): Title of the plot.
    xlabel (str): Label for the X-axis.
    ylabel (str): Label for the Y-axis.
    """
    plt.figure(figsize=(10, 8))

    # Create a mesh grid for pcolormesh
    X, Y = np.meshgrid(x_values, np.arange(len(y_values_list[0])))
    Z = np.array(phase_matrix)

    # Plot using pcolormesh
    plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    plt.colorbar(label='Phase')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
