import functools
import numpy as np
from typing import Dict
from tqdm.notebook import tqdm
from scipy.constants import hbar, e, k
from scipy.interpolate import UnivariateSpline
from utils import filter_args, plot_vs_parameters
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay, sigmax


@functools.lru_cache(maxsize=None)
def phi_zpf(Ec, El):
    return (8.0 * Ec / El) ** 0.25

@functools.lru_cache(maxsize=None)
def charge_number_operator(Ec, El, dimension) -> Qobj:
    return 1j * (destroy(dimension).dag() - destroy(dimension)) / phi_zpf(Ec, El) / np.sqrt(2)

@functools.lru_cache(maxsize=None)
def charge_number_operator_total(Ec, El, dimension) -> Qobj:
    return tensor(charge_number_operator(Ec, El, dimension), qeye(2))

@functools.lru_cache(maxsize=None)
def phase_operator(Ec, El, dimension) -> Qobj:
    return (destroy(dimension).dag() + destroy(dimension)) * phi_zpf(Ec, El) / np.sqrt(2)

@functools.lru_cache(maxsize=None)
def phase_operator_total(Ec, El, dimension) -> Qobj:
    return tensor(phase_operator(Ec, El, dimension), qeye(2))

@functools.lru_cache(maxsize=None)
def dHdr_operator(Ec, El, r, Delta, dimension, model = "zazunov") -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)

    if model == 'zazunov':
        dReZdr = 1/2*(r*phase_op*(r*phase_op/2).cosm()*(phase_op/2).sinm()+(-phase_op*(phase_op/2).cosm()+2*(phase_op/2).sinm())*(r*phase_op/2).sinm())
        dImZdr = -1/2*(r*phase_op/2).cosm()*(phase_op*(phase_op/2).cosm()-2*(phase_op/2).sinm())-1/2*r*phase_op*(phase_op/2).sinm()*(r*phase_op/2).sinm()
        return Delta*(tensor(dReZdr,sigmaz())+tensor(dImZdr,sigmay()))
    elif model == 'jrl':
        return -Delta*tensor((phase_op/2).sinm(),sigmay())

OPERATOR_FUNCTIONS = {
    'charge_number': charge_number_operator_total,
    'phase': phase_operator_total,
    'dHdr': dHdr_operator,
}

OPERATOR_LABELS = {
    'charge_number': r'\hat{n}',
    'phase': r'\hat{\varphi}',
    'dHdr': r'\hat{\partial H /\partial r}'
}

@functools.lru_cache(maxsize=None)
def ReZ(Ec, El, r: float, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return (phase_op/2).cosm()*(r*phase_op/2).cosm() + r*(phase_op/2).sinm()*(r*phase_op/2).sinm()
    # return (phase_op/2).cosm()

@functools.lru_cache(maxsize=None)
def ImZ(Ec, El, r, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    # return r*(phase_op/2).sinm()
    return -(phase_op/2).cosm()*(r*phase_op/2).sinm() + r*(phase_op/2).sinm()*(r*phase_op/2).cosm()

@functools.lru_cache(maxsize=None)
def jrl_potential(Ec, El, r, er, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return tensor((phase_op/2).cosm(),sigmax()) + r*tensor((phase_op/2).sinm(),sigmay()) - er*tensor(qeye(dimension),sigmaz())
    # return  tensor((phase_op/2).cosm(),sigmaz()) - er*tensor(qeye(dimension),sigmax()) + r* tensor((phase_op/2).sinm(),sigmay()) 

@functools.lru_cache(maxsize=None)
def delta(Ec, El, phi_ext, dimension):
    return phase_operator(Ec, El, dimension) - phi_ext

@functools.lru_cache(maxsize=None)
def hamiltonian(Ec, El, Delta, r, phi_ext: float, er=0, dimension = 100, model = 'ferbo') -> Qobj:
    charge_op = charge_number_operator(Ec, El, dimension)
    delta_val = delta(Ec, El, phi_ext, dimension)

    if model == 'zazunov':
        ReZ_val = ReZ(Ec, El, r, dimension)
        ImZ_val = ImZ(Ec, El, r, dimension)
        return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) + Delta*(tensor(ReZ_val, sigmaz()) + tensor(ImZ_val, sigmay()))
    elif model == 'jrl': #Josephson Resonance level
        return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) - Delta*jrl_potential(Ec, El, r, er, dimension)


def eigen_vs_parameter(parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=6, calculate_states=False, plot=True, filename=None, **kwargs):
    if parameter_name not in ["Ec", "El", "Delta", "phi_ext", "r", "er"]:
        raise ValueError("parameter_name must be one of the following: 'Ec', 'El', 'Delta', 'phi_ext', 'r', 'er'")
    
    eigenenergies = np.zeros((len(parameter_values), eigvals))
    eigenstates = [] if calculate_states else None

    params = fixed_params.copy()
    for i, param_value in enumerate(tqdm(parameter_values)):
        params[parameter_name] = param_value
        h = hamiltonian(**params)
        if calculate_states:
            energy, states = h.eigenstates(eigvals=eigvals)
            eigenenergies[i] = np.real(energy)
            eigenstates.append(states)
        else:
            eigenenergies[i] = np.real(h.eigenenergies(eigvals=eigvals))
    
    if plot:
        ylabel = 'Eigenenergies'
        # title = f'Eigenenergies vs {parameter_name}'
        title  = ', '.join([f'{key}={value}' for key, value in fixed_params.items()])
        plot_vs_parameters(parameter_values, eigenenergies, parameter_name, ylabel, title, filename, **kwargs)

    return (eigenenergies, eigenstates) if calculate_states else eigenenergies

def matrix_elements_vs_parameter(parameter_name: str, parameter_values, operator_name: str, fixed_params: Dict[str, float], state_i=0, state_j=1, plot=True, filename=None, **kwargs):
    # Asegúrate de que state_i y state_j estén cubiertos en los cálculos de eigen
    eigvals = max(state_i, state_j) + 1

    # Utiliza la función existente para obtener las eigenenergías y eigenestados
    eigenenergies, eigenstates = eigen_vs_parameter(parameter_name, parameter_values, fixed_params, eigvals=eigvals, calculate_states=True, plot=False)

    matrix_elements = np.zeros(len(parameter_values), dtype=complex)
    operator_function = OPERATOR_FUNCTIONS.get(operator_name)
    if operator_function is None:
        raise ValueError(f"Unknown operator name: {operator_name}")

    params = fixed_params.copy()
    for i, param_value in enumerate(parameter_values):
        params[parameter_name] = param_value
        filtered_params = filter_args(operator_function, params)

        operator_function = OPERATOR_FUNCTIONS.get(operator_name)
        if operator_function is None:
            raise ValueError(f"Unknown operator name: {operator_name}")

        operator = operator_function(**filtered_params)
        matrix_elements[i] = operator.matrix_element(eigenstates[i][state_i], eigenstates[i][state_j])

    if plot:
        operator_label = OPERATOR_LABELS.get(operator_name, operator_name)
        ylabel = rf'$|\langle {state_i} | {operator_label} | {state_j} \rangle|^2$'
        # title = rf'{ylabel} vs {parameter_name}'
        title  = ', '.join([f'{key}={value}' for key, value in fixed_params.items()])
        plot_vs_parameters(parameter_values, np.abs(matrix_elements)**2, parameter_name, ylabel, title, filename, **kwargs)

    return matrix_elements, eigenenergies


def derivative_eigenenergies(parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=2):

    energies = eigen_vs_parameter('phi_ext', parameter_values, fixed_params, eigvals,calculate_states = False, plot=False)

    # TODO: Make it general for any different transition
    energy_transition = energies[:, 1] - energies[:, 0]
  
    # Crear splines para suavizar y derivar las energías
    spline_01 = UnivariateSpline(parameter_values, energy_transition, k=4, s=0)

    # Calcular la primera y segunda derivada
    first_derivative_01 = spline_01.derivative(n=1)
    second_derivative_01 = spline_01.derivative(n=2)

    # Evaluar las derivadas en los valores del parámetro
    first_derivatives_01 = first_derivative_01(parameter_values)
    second_derivatives_01 = second_derivative_01(parameter_values)

    ylabel = [r'$\left | \partial f_{01}/\partial \varphi_{ext} \right |^2$',
              r'$\left | \partial^2 f_{01}/\partial \varphi_{ext}^2 \right | ^2$']
    plot_vs_parameters(parameter_values, [first_derivatives_01**2,second_derivatives_01**2], [parameter_name]*2, ylabel)

    return first_derivatives_01, second_derivatives_01

def t1_vs_parameter(parameter_name: str, parameter_values, operator_name, spectral_density, fixed_params: Dict[str, float], state_i = 0, state_j = 1, plot=True, filename=None):
    raise NotImplementedError
    # matrix_elements, eigenenergies = matrix_elements_vs_parameter(parameter_name, parameter_values, operator_name, fixed_params, state_i, state_j, plot=False)
    # t1 = hbar**2/np.abs(matrix_elements)**2/spectral_density(eigenenergies)
    # if plot:
    #     ylabel = f'T1'
    #     title = f'T1 vs {parameter_name}'
    #     plot_vs_parameters(parameter_values, t1, parameter_name, ylabel, title, filename)
    # return t1