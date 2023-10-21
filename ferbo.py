import inspect
import functools
import numpy as np
from typing import Dict
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay


@functools.lru_cache(maxsize=None)
def phi_zpf(Ec, El):
    return (8.0 * Ec / El) ** 0.25

@functools.lru_cache(maxsize=None)
def charge_number_operator(Ec, El, dimension) -> Qobj:
    return 1j * (destroy(dimension).dag() - destroy(dimension)) / phi_zpf(Ec, El) / np.sqrt(2)

@functools.lru_cache(maxsize=None)
def phase_operator(Ec, El, dimension) -> Qobj:
    return (destroy(dimension).dag() + destroy(dimension)) * phi_zpf(Ec, El) / np.sqrt(2)

OPERATOR_FUNCTIONS = {
    'charge_number': charge_number_operator,
    'phase': phase_operator,
}

@functools.lru_cache(maxsize=None)
def ReZ(Ec, El, r: float, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return (phase_op/2).cosm()*(r*phase_op/2).cosm() + r*(phase_op/2).sinm()*(r*phase_op/2).sinm()

@functools.lru_cache(maxsize=None)
def ImZ(Ec, El, r, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return -(phase_op/2).cosm()*(r*phase_op/2).sinm() + r*(phase_op/2).sinm()*(r*phase_op/2).cosm()

@functools.lru_cache(maxsize=None)
def delta(Ec, El, phi_ext, dimension):
    return phase_operator(Ec, El, dimension) - phi_ext

@functools.lru_cache(maxsize=None)
def hamiltonian(Ec, El, Delta, r, phi_ext: float, dimension = 100) -> Qobj:
    charge_op = charge_number_operator(Ec, El, dimension)
    ReZ_val = ReZ(Ec, El, r, dimension)
    ImZ_val = ImZ(Ec, El, r, dimension)
    delta_val = delta(Ec, El, phi_ext, dimension)
    return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) + Delta*(tensor(ReZ_val, sigmaz()) + tensor(ImZ_val, sigmay()))

def eigenenergies_vs_parameter(parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=6, plot = True, filename=None) -> np.ndarray:
    if parameter_name not in ["Ec", "El", "Delta", "phi_ext", "r"]:
            raise ValueError("parameter_name must be one of the following: 'Ec', 'El', 'Delta', 'phi_ext', 'r'")
    eigenenergies = np.zeros((len(parameter_values), eigvals))
    for i, param_value in enumerate(tqdm(parameter_values)):
        # Actualizar el valor del parámetro variable
        params = fixed_params.copy()
        params[parameter_name] = param_value
        # Calcular las autoenergías
        h = hamiltonian(**params)
        eigenenergies[i] = np.real(h.eigenenergies(eigvals=eigvals))
    
    if plot:
        ylabel = 'Eigenenergies'
        title = f'Eigenenergies vs {parameter_name}'
        plot_vs_parameter(parameter_values, eigenenergies, parameter_name, ylabel, title, filename)
    
    return eigenenergies

def matrix_elements_vs_parameter(parameter_name: str, parameter_values, operator_name: str,fixed_params: Dict[str, float], state_i = 0, state_j = 1, plot=True, filename=None):
    matrix_elements = np.zeros(len(parameter_values), dtype=complex)
    operator_function = OPERATOR_FUNCTIONS.get(operator_name)
    if operator_function is None:
        raise ValueError(f"Unknown operator name: {operator_name}")
    
    for k, param_value in enumerate(tqdm(parameter_values)):
        params = fixed_params.copy()
        params[parameter_name] = param_value
        filtered_params = filter_args(operator_function, params)
        h = hamiltonian(**params)
        eigenstates = h.eigenstates(eigvals=max(state_i, state_j) + 1)[1]
        operator = operator_function(**filtered_params)
        matrix_elements[k] = operator.matrix_element(eigenstates[state_i], eigenstates[state_j])

    if plot:
        ylabel = f'Matrix Elements of {operator_name}'
        title = f'Matrix Elements of {operator_name} vs {parameter_name}'
        plot_vs_parameter(parameter_values, np.abs(matrix_elements), parameter_name, ylabel, title, filename)

    return matrix_elements

def filter_args(func, params):
    """ Filtra un diccionario de argumentos para incluir solo los que necesita una función. """
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}

def plot_vs_parameter(x_values, y_values, parameter_name, ylabel, title=None, filename=None):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values)
    ax.set_xlabel(parameter_name)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if filename:
        fig.savefig(filename)
    plt.show()


