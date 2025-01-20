import numpy as np
from typing import Dict, List, Tuple, Union
from tqdm.notebook import tqdm
from scipy.constants import hbar, e, k
from scipy.interpolate import UnivariateSpline
from src.files_to_organize.utils import filter_args, plot_vs_parameters
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay, sigmax
# import dynamiqs as dq

class Ferbo:
    def __init__(self, Ec, El, Gamma, delta_Gamma, er, flux, dimension):
        self.Ec = Ec
        self.El = El
        self.Gamma = Gamma
        self.delta_Gamma = delta_Gamma
        self.er = er
        self.flux = flux
        self.dimension = dimension
    
    def charge_number_operator(self) -> Qobj:
        return 1j/2 * (self.El/2/self.Ec)**0.25 * (destroy(self.dimension).dag() - destroy(self.dimension))
    
    def phase_operator(self) -> Qobj:
        return (2*self.Ec/self.El)**0.25 * (destroy(self.dimension).dag() + destroy(self.dimension))
    
    def charge_number_operator_total(self) -> Qobj:
        return tensor(self.charge_number_operator(), qeye(2))
    
    def phase_operator_total(self) -> Qobj:
        return tensor(self.phase_operator(), qeye(2))
    
    def jrl_potential(self) -> Qobj:
        phase_op = self.phase_operator()
        return  - self.Gamma * tensor((phase_op/2).cosm(),sigmax()) - self.delta_Gamma * tensor((phase_op/2).sinm(),sigmay()) - self.er * tensor(qeye(self.dimension),sigmaz())
    
    def hamiltonian(self) -> Qobj:
        charge_term = 4 * self.Ec * self.charge_number_operator_total()**2
        inductive_term = 0.5 * self.El * (self.phase_operator_total() + self.flux)**2
        potential = self.jrl_potential()
        return charge_term + inductive_term + potential
    
    def get_spectrum_vs_paramvals(self, param_name: str, param_vals: List[float], evals_count: int = 6, subtract_ground: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        eigenenergies_array = []
        eigenstates_array = []
        
        for val in tqdm(param_vals):
            setattr(self, param_name, val)
            H = self.hamiltonian()
            eigenenergies, eigenstates = H.eigenstates(eigvals=evals_count)
            if subtract_ground:
                eigenenergies -= eigenenergies[0]
            eigenenergies_array.append(eigenenergies)
            eigenstates_array.append(eigenstates)
        
        return np.array(eigenenergies_array), np.array(eigenstates_array)
    
    def matrixelement_table(self, operator: str, evecs: np.ndarray = None, evals_count: int = 6) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Returns a table of matrix elements for a given operator with respect to the eigenstates.

        Parameters
        ----------
        operator : str
            The name of the operator.
        evecs : np.ndarray, optional
            The eigenstates (default is None, in which case they are calculated).
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).

        Returns
        -------
        np.ndarray
            The table of matrix elements.
        """
        
        if evecs is None:
            H = self.hamiltonian()
            _, evecs = H.eigenstates(eigvals=evals_count)
            
        operator_matrix = getattr(self, operator)()
        
        evecs = np.array([evec.full().flatten() for evec in evecs])
        evecs_dag = np.conjugate(evecs.T)

        if isinstance(operator_matrix, Qobj):
            operator_matrix = operator_matrix.full()
            
        matrix_elements = evecs @ operator_matrix @ evecs_dag
        return matrix_elements
    
    def get_matelements_vs_paramvals(self, operators: Union[str, List[str]], param_name: str, param_vals: np.ndarray, evals_count: int = 6) -> Dict[str, Dict[str, np.ndarray]]:
        if isinstance(operators, str):
            operators = [operators]
        
        eigenenergies_array, eigenstates_array = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=evals_count, subtract_ground=False)
        paramvals_count = len(param_vals)
        results = {
            'param_vals': param_vals,
            'eigenenergies': eigenenergies_array
        }
        
        for operator in operators:
            results[operator] = np.empty((paramvals_count, evals_count, evals_count), dtype=np.complex_)
        
        for index, paramval in enumerate(param_vals):
            evecs = eigenstates_array[index]
            for operator in operators:
                results[operator][index] = self.matrixelement_table(operator, evecs=evecs, evals_count=evals_count)
        
        return results
    
    def t1_capacitive(self, i: int = 1, j: int = 0, Q_cap: Union[float, Callable] = None, T: float = 0.015, esys: Tuple[np.ndarray, np.ndarray] = None, get_rate: bool = False, noise_op: Optional[Union[np.ndarray, Qobj]] = None) -> float:
        
        if Q_cap is None:
            Q_cap_fun = lambda omega: 1e6 * (2 * np.pi * 6e9 / omega)**0.7
        elif callable(Q_cap):
            Q_cap_fun = Q_cap
        else:
            Q_cap_fun = lambda omega: Q_cap

        def spectral_density(omega, T):
            return 32 * np.pi * (self.Ec * 1e9) / Q_cap_fun(omega) * 1/np.tanh(hbar * omega / (2 * k * T))

        noise_op = noise_op or self.n_operator_total()
        if isinstance(noise_op, Qobj):
            noise_op = noise_op.full()
            
        H = self.hamiltonian()
        evals, evecs = H.eigenstates(eigvals=max(i, j) + 1) if esys is None else esys
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        
        s = spectral_density(omega, T)
        matrix_elements = self.matrixelement_table('n_operator_total', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = matrix_element**2 * s
        return rate if get_rate else 1 / rate
    
    def t1_inductive(self, i: int = 1, j: int = 0, Q_ind: float = 500e6, T: float = 0.015, esys: Tuple[np.ndarray, np.ndarray] = None, get_rate: bool = False, noise_op: Optional[Union[np.ndarray, Qobj]] = None) -> float:
        def spectral_density(omega, T):
            return 4 * np.pi * (self.El * 1e9) / Q_ind * 1 / np.tanh(hbar * omega / (2 * k * T))

        noise_op = noise_op or self.phase_operator_total()
        if isinstance(noise_op, Qobj):
            noise_op = noise_op.full()
            
        H = self.hamiltonian()
        evals, evecs = H.eigenstates(eigvals=max(i, j) + 1) if esys is None else esys
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        s = spectral_density(omega, T)
        
        matrix_elements = self.matrixelement_table('phase_operator_total', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = matrix_element**2 * s
        return rate if get_rate else 1 / rate


##### Revisit this functions later ######

# def  hamiltonian(Ec, El, Delta, r, phi_ext: float, er=0, dimension = 100, model = 'jrl') -> Qobj:
#     charge_op = charge_number_operator(Ec, El, dimension)
#     delta_val = delta(Ec, El, phi_ext, dimension)

#     if model == 'zazunov':
#         ReZ_val = ReZ(Ec, El, r, dimension)
#         ImZ_val = ImZ(Ec, El, r, dimension)
#         return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) - Delta*(tensor(ReZ_val, sigmaz()) + tensor(ImZ_val, sigmay()))
#     elif model == 'jrl': #Josephson Resonance level
#         return 4*Ec*tensor(charge_op**2, qeye(2)) + 0.5*El*tensor(delta_val**2, qeye(2)) - Delta*jrl_hamiltonian(Ec, El, r, er, dimension)

# def dHdr_operator(Ec, El, r, Delta, dimension, model = "jrl") -> Qobj:
#     phase_op = phase_operator(Ec, El, dimension)

#     if model == 'zazunov':
#         dReZdr = 1/2*(r*phase_op*(r*phase_op/2).cosm()*(phase_op/2).sinm()+(-phase_op*(phase_op/2).cosm()+2*(phase_op/2).sinm())*(r*phase_op/2).sinm())
#         dImZdr = -1/2*(r*phase_op/2).cosm()*(phase_op*(phase_op/2).cosm()-2*(phase_op/2).sinm())-1/2*r*phase_op*(phase_op/2).sinm()*(r*phase_op/2).sinm()
#         return Delta*(tensor(dReZdr,sigmaz())+tensor(dImZdr,sigmay()))
#     elif model == 'jrl':
#         return -Delta*tensor((phase_op/2).sinm(),sigmay())
    
# def dHder_operator(Delta, dimension, model = "jrl") -> Qobj:
#     if model == 'zazunov':
#         raise ValueError(f'Not valid for the Zazunov model.')
#     elif model == 'jrl':
#         return - Delta*tensor(qeye(dimension),sigmax())

# OPERATOR_FUNCTIONS = {
#     'charge_number': charge_number_operator_total,
#     'phase': phase_operator_total,
#     'dHdr': dHdr_operator,
#     'dHder': dHder_operator,
# }

# OPERATOR_LABELS = {
#     'charge_number': r'\hat{n}',
#     'phase': r'\hat{\varphi}',
#     'dHdr': r'\hat{\partial H /\partial r}',
#     'dHder': r'\hat{\partial H /\partial \varepsilon_r}'
# }

# def ReZ(Ec, El, r: float, dimension) -> Qobj:
#     phase_op = phase_operator(Ec, El, dimension)
#     return (phase_op/2).cosm()*(r*phase_op/2).cosm() + r*(phase_op/2).sinm()*(r*phase_op/2).sinm()
#     # return (phase_op/2).cosm()

# def ImZ(Ec, El, r, dimension) -> Qobj:
#     phase_op = phase_operator(Ec, El, dimension)
#     return -(phase_op/2).cosm()*(r*phase_op/2).sinm() + r*(phase_op/2).sinm()*(r*phase_op/2).cosm()
#     # return r*(phase_op/2).sinm()

# def ferbo_potential(phi, El, Delta):
#     ground_potential = 0.5*El*phi**2 - Delta*np.cos(phi/2) 
#     excited_potential = 0.5*El*phi**2 + Delta*np.cos(phi/2) 
#     return  ground_potential, excited_potential

# def delta(Ec, El, phi_ext, dimension):
#     return phase_operator(Ec, El, dimension) - phi_ext


# def eigen_vs_parameter(parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=6, calculate_states=False, plot=True, filename=None, **kwargs):
#     if parameter_name not in ["Ec", "El", "Delta", "phi_ext", "r", "er"]:
#         raise ValueError("parameter_name must be one of the following: 'Ec', 'El', 'Delta', 'phi_ext', 'r', 'er'")
    
#     eigenenergies = np.zeros((len(parameter_values), eigvals))
#     eigenstates = [] if calculate_states else None

#     params = fixed_params.copy()
#     # for i, param_value in enumerate(tqdm(parameter_values)):
#     for i, param_value in enumerate(parameter_values):

#         params[parameter_name] = param_value
#         h = hamiltonian(**params)
#         if calculate_states:
#             energy, states = h.eigenstates(eigvals=eigvals)
#             eigenenergies[i] = np.real(energy)
#             eigenstates.append(states)
#         else:
#             eigenenergies[i] = np.real(h.eigenenergies(eigvals=eigvals))
    
#     if plot:
#         ylabel = 'Eigenenergies'
#         # title = f'Eigenenergies vs {parameter_name}'
#         title  = ', '.join([f'{key}={value}' for key, value in fixed_params.items()])
#         plot_vs_parameters(parameter_values, eigenenergies, parameter_name, ylabel, title, filename, **kwargs)

#     return (eigenenergies, eigenstates) if calculate_states else eigenenergies

# def matrix_elements_vs_parameter(parameter_name: str, parameter_values, operator_name: str, fixed_params: Dict[str, float], state_i=0, state_j=1, plot=True, filename=None, **kwargs):
#     # Asegúrate de que state_i y state_j estén cubiertos en los cálculos de eigen
#     eigvals = kwargs.get('eigvals', max(state_i, state_j) + 1)

#     # Utiliza la función existente para obtener las eigenenergías y eigenestados
#     eigenenergies, eigenstates = eigen_vs_parameter(parameter_name, parameter_values, fixed_params, eigvals=eigvals, calculate_states=True, plot=False)

#     matrix_elements = np.zeros(len(parameter_values), dtype=complex)
#     operator_function = OPERATOR_FUNCTIONS.get(operator_name)
#     if operator_function is None:
#         raise ValueError(f"Unknown operator name: {operator_name}")

#     params = fixed_params.copy()
#     for i, param_value in enumerate(parameter_values):
#         params[parameter_name] = param_value
#         filtered_params = filter_args(operator_function, params)

#         operator_function = OPERATOR_FUNCTIONS.get(operator_name)
#         if operator_function is None:
#             raise ValueError(f"Unknown operator name: {operator_name}")

#         operator = operator_function(**filtered_params)
#         matrix_elements[i] = operator.matrix_element(eigenstates[i][state_i], eigenstates[i][state_j])

#     if plot:
#         operator_label = OPERATOR_LABELS.get(operator_name, operator_name)
#         ylabel = rf'$|\langle {state_i} | {operator_label} | {state_j} \rangle|^2$'
#         # title = rf'{ylabel} vs {parameter_name}'
#         title  = ', '.join([f'{key}={value}' for key, value in fixed_params.items()])
#         plot_vs_parameters(parameter_values, np.abs(matrix_elements)**2, parameter_name, ylabel, title, filename, **kwargs)

#     return matrix_elements, eigenenergies


# def derivative_eigenenergies(external_param, parameter_name, parameter_values, fixed_params: Dict[str, float], eigvals=2, plot=True):

#     ext_name = OPERATOR_LABELS.get(external_param, r'\lambda')
#     ylabel = [rf'$\left | \partial f_{{01}}/\partial {ext_name} \right |^2$',
#               rf'$\left | \partial^2 f_{{01}}/\partial {ext_name}^2 \right | ^2$']
    
#     if parameter_name == external_param and parameter_name in ['phi_ext','r','er']:
#         energies = eigen_vs_parameter(parameter_name, parameter_values, fixed_params, eigvals,calculate_states = False, plot=False)
#         energy_transition = energies[:, 1] - energies[:, 0]
#         spline_01 = UnivariateSpline(parameter_values, energy_transition, k=4, s=0)
#         first_derivative_01 = spline_01.derivative(n=1)
#         second_derivative_01 = spline_01.derivative(n=2)

#         df01_dextparam = first_derivative_01(parameter_values)
#         d2f01_dextparam2 = second_derivative_01(parameter_values)

#     else:
#         h = 0.001
#         aux_fixed_params = fixed_params.copy()
#         energies_center = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
#         aux_fixed_params[external_param] = fixed_params[external_param] - h
#         energies_lower = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
#         aux_fixed_params[external_param] = fixed_params[external_param] + h
#         energies_higher = eigen_vs_parameter(parameter_name, parameter_values, aux_fixed_params, eigvals,calculate_states = False, plot=False)
#         f01_center = energies_center[:,1] - energies_center[:,0]
#         f01_higher = energies_higher[:,1] - energies_higher[:,0]
#         f01_lower = energies_lower[:,1] - energies_lower[:,0]

#         df01_dextparam = (f01_higher - f01_lower)/2/h
#         d2f01_dextparam2 = (f01_higher - 2*f01_center + f01_lower)/h**2

#     if plot:
#         plot_vs_parameters(parameter_values, [df01_dextparam,d2f01_dextparam2], [parameter_name]*2, ylabel)

#     return df01_dextparam,d2f01_dextparam2


# def t1_vs_parameter(parameter_name: str, parameter_values, operator_name, spectral_density, fixed_params: Dict[str, float], state_i = 0, state_j = 1, plot=True, filename=None):
    raise NotImplementedError
    # matrix_elements, eigenenergies = matrix_elements_vs_parameter(parameter_name, parameter_values, operator_name, fixed_params, state_i, state_j, plot=False)
    # t1 = hbar**2/np.abs(matrix_elements)**2/spectral_density(eigenenergies)
    # if plot:
    #     ylabel = f'T1'
    #     title = f'T1 vs {parameter_name}'
    #     plot_vs_parameters(parameter_values, t1, parameter_name, ylabel, title, filename)
    # return t1