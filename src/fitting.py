import pandas as pd
import numpy as np
import scqubits as sq
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm
from IPython.utils import io

def fit_fluxonium(transition_files, bounds):
    """
    Fits the Fluxonium model based on the provided transitions.

    Parameters:
    -----------
    transition_files : list of tuples
        Each tuple contains:
        - file_path (str): Path to the CSV file containing the transition data.
        - i (int): Initial energy level index.
        - j (int): Final energy level index.
        The CSV files should have columns 'phi' and 'e_ij'.
        
    bounds : dict
        Dictionary with keys 'EJ', 'EC', 'EL' defining the bounds for each parameter.
        Example: {'EJ': (0.8, 15), 'EC': (1.2, 2), 'EL': (2, 5)}

    Returns:
    --------
    fluxonium_fit : scqubits.Fluxonium
        The fitted Fluxonium object.
    result : OptimizeResult
        The result of the optimization.
    
    Example:
    --------
    transition_files = [
        ('../data/data_fitting/E01.csv', 0, 1),  # transition from 0 to 1
        ('../data/data_fitting/E02.csv', 1, 2)   # transition from 1 to 2
    ]

    bounds = {
        'EJ': (0.8, 15),  # EJ bounds
        'EC': (1.2, 2),   # EC bounds
        'EL': (2, 5)      # EL bounds
    }

    fluxonium_fit, result = fit_fluxonium(transition_files, bounds)
    """
    
    transitions = []
    
    for file_path, i, j in transition_files:
        df = pd.read_csv(file_path, delimiter=';', decimal=',', header=None, names=['phi', 'e_ij'])
        transitions.append((df, i, j))

    def minimizer(normalized_params):
        # Unpack and denormalize the parameters
        norm_EJ, norm_EC, norm_EL = normalized_params
        EJ = norm_EJ * (bounds['EJ'][1] - bounds['EJ'][0]) + bounds['EJ'][0]
        EC = norm_EC * (bounds['EC'][1] - bounds['EC'][0]) + bounds['EC'][0]
        EL = norm_EL * (bounds['EL'][1] - bounds['EL'][0]) + bounds['EL'][0]

        fluxonium = sq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=0, cutoff=40)
        
        cost = 0
        n_points = 0
        
        for transition in transitions:
            phi = transition[0]['phi'].values
            e_ij_exp = transition[0].iloc[:, 1].values
            i, j = transition[1], transition[2]
            
            with io.capture_output() as captured:
                val = fluxonium.get_spectrum_vs_paramvals(param_name='flux', param_vals=phi, evals_count=max(i, j) + 1, subtract_ground=False)
            e_i = val.energy_table[:, i]
            e_j = val.energy_table[:, j]
            e_ij = e_j - e_i
            
            cost += np.sum((e_ij - e_ij_exp) ** 2)
            n_points += len(phi)
        
        cost_function = np.sqrt(cost) / n_points
        return cost_function

    progress = []

    # Create the progress bar
    progress_bar = tqdm(total=100, desc="Progress")

    # Define the callback function with convergence tracking
    def callback(params, convergence):
        # Register the current state of the fit
        progress.append(params)
        # Calculate the progress based on convergence
        progress_percentage = min(100, int((1 - convergence) * 100))
        # Update the progress bar
        progress_bar.n = progress_percentage
        progress_bar.refresh()
        # Display current parameters and convergence
        print(f"Parameters: {params}, Convergence: {convergence}")
        if convergence <= 1e-6:  # Adjust the convergence threshold as needed
            progress_bar.close()

    # Normalize bounds to [0, 1] for optimization
    normalized_bounds = [(0, 1), (0, 1), (0, 1)]

    # Execute the optimization
    result = differential_evolution(func=minimizer, bounds=normalized_bounds, callback=callback, disp=False, maxiter=100)

    # Denormalize the result
    norm_EJ, norm_EC, norm_EL = result.x
    EJ = norm_EJ * (bounds['EJ'][1] - bounds['EJ'][0]) + bounds['EJ'][0]
    EC = norm_EC * (bounds['EC'][1] - bounds['EC'][0]) + bounds['EC'][0]
    EL = norm_EL * (bounds['EL'][1] - bounds['EL'][0]) + bounds['EL'][0]

    fluxonium_fit = sq.Fluxonium(EJ, EC, EL, flux=0, cutoff=40)

    return fluxonium_fit, result