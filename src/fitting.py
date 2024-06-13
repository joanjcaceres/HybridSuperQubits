import pandas as pd
import numpy as np
import scqubits as sq
from scipy.optimize import differential_evolution
from tqdm.notebook import tqdm
from IPython.utils import io
import matplotlib.pyplot as plt

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
    progress_bar = tqdm(total=100, desc="Iterations")

    # Define the callback function
    def callback(params, convergence):
        # Register the current state of the fit
        progress.append(params)
        # Update the progress bar
        progress_bar.update(len(progress) - progress_bar.n)
        if len(progress) >= 100:
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


def get_transitions_from_levels(fluxonium_fit:sq.Fluxonium, flux_array, evals_count=6, base_levels=None):
    """
    Gets specific energy transitions from base levels to higher levels for a fitted Fluxonium object.

    Parameters:
    -----------
    fluxonium_fit : scqubits.Fluxonium
        The fitted Fluxonium object.
    flux_array : numpy.ndarray
        Array of flux values.
    evals_count : int, optional
        Number of eigenvalues to calculate (default is 6).
    base_levels : list of int, optional
        List of base levels from which to calculate transitions.

    Returns:
    --------
    flux_array : numpy.ndarray
        Array of flux values.
    transition_dict : dict
        Dictionary of calculated transitions with keys as tuples (i, j).
    """
    fluxspec = fluxonium_fit.get_spectrum_vs_paramvals(
        param_name='flux',
        param_vals=flux_array,
        evals_count=evals_count,
        subtract_ground=False
    )
    evals_matrix = fluxspec.energy_table
    transition_dict = {}

    if base_levels is None:
        base_levels = [0]

    for i in base_levels:
        transitions = evals_matrix - evals_matrix[:, i].reshape(-1, 1)
        for j in range(i + 1, evals_count):  # Ensure j > i
            transition_dict[(i, j)] = transitions[:, j]

    return flux_array, transition_dict

def plot_transitions(flux_array, transition_dict, fig, ax, fluxonium_fit, **kwargs):
    """
    Plots specific energy transitions on an existing figure.

    Parameters:
    -----------
    flux_array : numpy.ndarray
        Array of flux values.
    transition_dict : dict
        Dictionary of calculated transitions with keys as tuples (i, j).
    fig : matplotlib.figure.Figure
        Existing figure to plot on.
    ax : matplotlib.axes._subplots.AxesSubplot
        Existing axis to plot on.
    fluxonium_fit : scqubits.Fluxonium
        The fitted Fluxonium object.
    **kwargs : dict
        Additional keyword arguments for plotting (e.g., alpha, linestyle, linewidth).

    """
    fig.suptitle(f'EJ = {np.round(fluxonium_fit.EJ, 2)}, EC = {np.round(fluxonium_fit.EC, 2)}, EL = {np.round(fluxonium_fit.EL, 3)}')
    alpha = kwargs.get('alpha', 0.5)
    linestyle = kwargs.get('linestyle', '--')
    linewidth = kwargs.get('linewidth', 0.8)

    # Get a colormap with enough unique colors
    cmap = plt.cm.get_cmap('tab10', len(transition_dict))

    for idx, ((i, j), transition) in enumerate(transition_dict.items()):
        color = cmap(idx)
        ax.plot(flux_array, transition, linestyle=linestyle, linewidth=linewidth, color=color, alpha=alpha, label=f'{i}->{j}')

    ax.legend()
    fig.tight_layout()
    plt.show()