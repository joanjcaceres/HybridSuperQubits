import functools
import numpy as np
from typing import Dict
from tqdm.notebook import tqdm
from scipy.constants import hbar, e, k
from scipy.interpolate import UnivariateSpline
from src.files_to_organize.utils import filter_args, plot_vs_parameters
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
def dHdr_operator(Ec, El, r, Delta, dimension, model = "jrl") -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)

    if model == 'zazunov':
        dReZdr = 1/2*(r*phase_op*(r*phase_op/2).cosm()*(phase_op/2).sinm()+(-phase_op*(phase_op/2).cosm()+2*(phase_op/2).sinm())*(r*phase_op/2).sinm())
        dImZdr = -1/2*(r*phase_op/2).cosm()*(phase_op*(phase_op/2).cosm()-2*(phase_op/2).sinm())-1/2*r*phase_op*(phase_op/2).sinm()*(r*phase_op/2).sinm()
        return Delta*(tensor(dReZdr,sigmaz())+tensor(dImZdr,sigmay()))
    elif model == 'jrl':
        return -Delta*tensor((phase_op/2).sinm(),sigmay())
    
@functools.lru_cache(maxsize=None)
def dHder_operator(Delta, dimension, model = "jrl") -> Qobj:
    if model == 'zazunov':
        raise ValueError(f'Not valid for the Zazunov model.')
    elif model == 'jrl':
        return - Delta*tensor(qeye(dimension),sigmax())

OPERATOR_FUNCTIONS = {
    'charge_number': charge_number_operator_total,
    'phase': phase_operator_total,
    'dHdr': dHdr_operator,
    'dHder': dHder_operator,
}

OPERATOR_LABELS = {
    'charge_number': r'\hat{n}',
    'phase': r'\hat{\varphi}',
    'dHdr': r'\hat{\partial H /\partial r}',
    'dHder': r'\hat{\partial H /\partial \varepsilon_r}'
}

@functools.lru_cache(maxsize=None)
def ReZ(Ec, El, r: float, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return (phase_op/2).cosm()*(r*phase_op/2).cosm() + r*(phase_op/2).sinm()*(r*phase_op/2).sinm()
    # return (phase_op/2).cosm()

@functools.lru_cache(maxsize=None)
def ImZ(Ec, El, r, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    return -(phase_op/2).cosm()*(r*phase_op/2).sinm() + r*(phase_op/2).sinm()*(r*phase_op/2).cosm()
    # return r*(phase_op/2).sinm()

@functools.lru_cache(maxsize=None)
def jrl_hamiltonian(Ec, El, r, er, dimension) -> Qobj:
    phase_op = phase_operator(Ec, El, dimension)
    # return tensor((phase_op/2).cosm(),sigmax()) + r*tensor((phase_op/2).sinm(),sigmay()) - er*tensor(qeye(dimension),sigmaz())
    return  tensor((phase_op/2).cosm(),sigmaz()) - er*tensor(qeye(dimension),sigmax()) + r* tensor((phase_op/2).sinm(),sigmay()) 

def ferbo_potential(phi, El, Delta):
    ground_potential = 0.5*El*phi**2 - Delta*np.cos(phi/2) 
    excited_potential = 0.5*El*phi**2 + Delta*np.cos(phi/2) 
    return  ground_potential, excited_potential

@functools.lru_cache(maxsize=None)
def delta(Ec, El, phi_ext, dimension):
    return phase_operator(Ec, El, dimension) - phi_ext

@functools.lru_cache(maxsize=None)
def  hamiltonian(Ec, El, Delta, r, phi_ext: float, er=0, dimension = 100, model = 'jrl') -> Qobj:
    charge_op = charge_number_operator(Ec, El, dimension)
    delta_val = delta(Ec, El, phi_ext, dimension)

    if model == 'zazunov':
        ReZ_val = ReZ(Ec, El, r, dimension)
        ImZ_val = ImZ(Ec, El, r, dimension)
        return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) - Delta*(tensor(ReZ_val, sigmaz()) + tensor(ImZ_val, sigmay()))
    elif model == 'jrl': #Josephson Resonance level
        return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) - Delta*jrl_hamiltonian(Ec, El, r, er, dimension)


def eigen_vs_parameter(parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=6, calculate_states=False, plot=True, filename=None, **kwargs):
    if parameter_name not in ["Ec", "El", "Delta", "phi_ext", "r", "er"]:
        raise ValueError("parameter_name must be one of the following: 'Ec', 'El', 'Delta', 'phi_ext', 'r', 'er'")
    
    eigenenergies = np.zeros((len(parameter_values), eigvals))
    eigenstates = [] if calculate_states else None

    params = fixed_params.copy()
    # for i, param_value in enumerate(tqdm(parameter_values)):
    for i, param_value in enumerate(parameter_values):

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
    eigvals = kwargs.get('eigvals', max(state_i, state_j) + 1)

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


def derivative_eigenenergies(external_param, parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=2, plot=True):

    ext_name = OPERATOR_LABELS.get(external_param, r'\lambda')
    ylabel = [rf'$\left | \partial f_{{01}}/\partial {ext_name} \right |^2$',
              rf'$\left | \partial^2 f_{{01}}/\partial {ext_name}^2 \right | ^2$']
    
    if parameter_name == external_param and parameter_name in ['phi_ext','r','er']:
        energies = eigen_vs_parameter(parameter_name, parameter_values, fixed_params, eigvals,calculate_states = False, plot=False)
        energy_transition = energies[:, 1] - energies[:, 0]
        spline_01 = UnivariateSpline(parameter_values, energy_transition, k=4, s=0)
        first_derivative_01 = spline_01.derivative(n=1)
        second_derivative_01 = spline_01.derivative(n=2)

        df01_dextparam = first_derivative_01(parameter_values)
        d2f01_dextparam2 = second_derivative_01(parameter_values)

    else:
        h = 0.001
        aux_fixed_params = fixed_params.copy()
        energies_center = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
        aux_fixed_params[external_param] = fixed_params[external_param] - h
        energies_lower = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
        aux_fixed_params[external_param] = fixed_params[external_param] + h
        energies_higher = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
        f01_center = energies_center[:,1] - energies_center[:,0]
        f01_higher = energies_higher[:,1] - energies_higher[:,0]
        f01_lower = energies_lower[:,1] - energies_lower[:,0]

        df01_dextparam = (f01_higher - f01_lower)/2/h
        d2f01_dextparam2 = (f01_higher - 2*f01_center + f01_lower)/h**2

    if plot:
        plot_vs_parameters(parameter_values, [df01_dextparam,d2f01_dextparam2], [parameter_name]*2, ylabel)

    return df01_dextparam,d2f01_dextparam2


def t1_vs_parameter(parameter_name: str, parameter_values, operator_name, spectral_density, fixed_params: Dict[str, float], state_i = 0, state_j = 1, plot=True, filename=None):
    raise NotImplementedError
    # matrix_elements, eigenenergies = matrix_elements_vs_parameter(parameter_name, parameter_values, operator_name, fixed_params, state_i, state_j, plot=False)
    # t1 = hbar**2/np.abs(matrix_elements)**2/spectral_density(eigenenergies)
    # if plot:
    #     ylabel = f'T1'
    #     title = f'T1 vs {parameter_name}'
    #     plot_vs_parameters(parameter_values, t1, parameter_name, ylabel, title, filename)
    # return t1