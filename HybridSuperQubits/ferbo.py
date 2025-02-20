import matplotlib.pyplot as plt
import numpy as np
from .qubit_base import QubitBase
from scipy.linalg import cosm, sinm, eigh
from typing import Any, Dict, Optional, Tuple, Union, Iterable
from .operators import destroy, creation, sigma_z, sigma_y, sigma_x

class Ferbo(QubitBase):
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'El': r'$E_L$',
        'Gamma': r'$\Gamma$',
        'delta_Gamma': r'$\delta \Gamma$',
        'er': r'$\epsilon_r$',
        'phase': r'$\Phi_{\mathrm{ext}} / \Phi_0$'
    }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'phase_operator': r'\hat{\phi}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    'd_hamiltonian_d_phase': r'\partial \hat{H} / \partial \phi_{{ext}}',
    'd_hamiltonian_d_EL': r'\partial \hat{H} / \partial E_L',
    'd_hamiltonian_d_deltaGamma': r'\partial \hat{H} / \partial \delta \Gamma',
    'd_hamiltonian_d_er': r'\partial \hat{H} / \partial \epsilon_r',
    }
    
    def __init__(self, Ec, El, Gamma, delta_Gamma, er, phase, dimension, flux_grouping: str = 'ABS'):
        """
        Initializes the Ferbo class with the given parameters.

        Parameters
        ----------
        Ec : float
            Charging energy.
        El : float
            Inductive energy.
        Gamma : float
            Coupling strength.
        delta_Gamma : float
            Coupling strength difference.
        er : float
            Energy relaxation rate.
        phase : float
            External magnetic phase.
        dimension : int
            Dimension of the Hilbert space.
        flux_grouping : str, optional
            Flux grouping ('L' or 'ABS') (default is 'L').
        """
        if flux_grouping not in ['L', 'ABS']:
            raise ValueError("Invalid flux grouping; must be 'L' or 'ABS'.")
        
        self.Ec = Ec
        self.El = El
        self.Gamma = Gamma
        self.delta_Gamma = delta_Gamma
        self.er = er
        self.phase = phase
        self.dimension = dimension // 2 * 2
        self.flux_grouping = flux_grouping
        super().__init__(self.dimension)
        
    @property
    def phase_zpf(self) -> float:
        """
        Returns the zero-point fluctuation of the phase.

        Returns
        -------
        float
            Zero-point fluctuation of the phase.
        """
        return (2 * self.Ec / self.El) ** 0.25
    
    @property
    def n_zpf(self) -> float:
        """
        Returns the zero-point fluctuation of the charge number.

        Returns
        -------
        float
            Zero-point fluctuation of the charge number.
        """
        return 1/2 * (self.El / 2 / self.Ec) ** 0.25
    
    def phi_osc(self) -> float:
        """
        Returns the oscillator length for the LC oscillator composed of the inductance and capacitance.

        Returns
        -------
        float
            Oscillator length.
        """
        return (8.0 * self.Ec / self.El) ** 0.25
    
    def n_operator(self) -> np.ndarray:
        """
        Returns the charge number operator.

        Returns
        -------
        np.ndarray
            The charge number operator.
        """
        single_mode_n_operator = 1j * self.n_zpf * (creation(self.dimension //2 ) - destroy(self.dimension // 2))
        return np.kron(single_mode_n_operator, np.eye(2))
    
    def phase_operator(self) -> np.ndarray:
        """
        Returns the total phase operator.

        Returns
        -------
        np.ndarray
            The total phase operator.
        """
        single_mode_phase_operator = self.phase_zpf * (creation(self.dimension //2) + destroy(self.dimension //2))
        return np.kron(single_mode_phase_operator, np.eye(2))        
    
    def jrl_potential(self) -> np.ndarray:
        """
        Returns the Josephson Resonance Level potential.

        Returns
        -------
        np.ndarray
            The Josephson Resonance Level potential.
        """
        phase_op = self.phase_operator()[::2, ::2]
        if self.flux_grouping == 'ABS':
            phase_op -= self.phase * np.eye(self.dimension // 2)
        
        return - self.Gamma * np.kron(cosm(phase_op/2), sigma_z()) - self.delta_Gamma * np.kron(sinm(phase_op/2), sigma_y()) + self.er * np.kron(np.eye(self.dimension // 2), sigma_x())
    
    # def zazunov_potential(self) -> np.ndarray:
        
    def hamiltonian(self) -> np.ndarray:
        """
        Returns the Hamiltonian of the system.

        Returns
        -------
        np.ndarray
            The Hamiltonian of the system.
        """
        charge_term = 4 * self.Ec * np.dot(self.n_operator(), self.n_operator())
        if self.flux_grouping == 'ABS':
            inductive_term = 0.5 * self.El * np.dot(self.phase_operator(), self.phase_operator())
        else:
            inductive_term = 0.5 * self.El * np.dot(self.phase_operator() + self.phase * np.eye(self.dimension), self.phase_operator() + self.phase * np.eye(self.dimension))
        potential = self.jrl_potential()
        return charge_term + inductive_term + potential
    
    def d_hamiltonian_d_EL(self) -> np.ndarray:
        
        if self.flux_grouping == 'L':
            phase_op = self.phase_operator()
        elif self.flux_grouping == 'ABS':
            phase_op = self.phase_operator() - self.phase * np.eye(self.dimension)

        return 1/2 * np.dot(phase_op, phase_op)
    
    def d_hamiltonian_d_ng(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the number of charge offset.
        
        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the number of charge offset.
        
        """
        return 8 * self.Ec * self.n_operator()
    
    def d_hamiltonian_d_phase(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the external magnetic phase.

        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the external magnetic phase.
        """
        if self.flux_grouping == 'L':
            return self.El * (self.phase_operator() + self.phase * np.eye(self.dimension))
        elif self.flux_grouping == 'ABS':
            phase_op = self.phase_operator()[::2,::2] - self.phase * np.eye(self.dimension // 2)
            return - self.Gamma/2 * np.kron(sinm(phase_op/2),sigma_z()) + self.delta_Gamma/2 * np.kron(cosm(phase_op/2),sigma_y())
                
    def d_hamiltonian_d_er(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the energy relaxation rate.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the energy relaxation rate.
        """
        return + np.kron(np.eye(self.dimension // 2),sigma_x())
    
    def d_hamiltonian_d_deltaGamma(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the coupling strength difference.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the coupling strength difference.
        """
        if self.flux_grouping == 'L':
            phase_op = self.phase_operator()[::2,::2]
        else:
            phase_op = self.phase_operator()[::2,::2] - self.phase * np.eye(self.dimension // 2)
        return - np.kron(sinm(phase_op/2),sigma_y())
    
    def wavefunction(
        self, 
        which: int = 0,
        phi_grid: np.ndarray = None,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        basis: str = 'phase',
        rotate: str = False,
        ) -> Dict[str, Any]:
        """
        Returns a wave function in the phi basis.

        Parameters
        ----------
        which : int, optional
            Index of desired wave function (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.
        basis : str, optional
            Basis in which to return the wavefunction ('phase' or 'charge') (default is 'phase').
        rotate : bool, optional
            Whether to rotate the basis (default is False).
        Returns
        -------
        Dict[str, Any]
            Wave function data containing basis labels, amplitudes, and energy.
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys
            
        dim = self.dimension//2
        
        if rotate:
            U = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
            change_of_basis_operator = np.kron(np.eye(dim), U)
            evecs = (change_of_basis_operator @ evecs)
        
        evecs = evecs.T
        
        if basis == 'phase':
            l_osc = self.phase_zpf
        elif basis == 'charge':
            l_osc = self.n_zpf     
                                
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[which, :]
        phi_wavefunc_amplitudes = np.zeros((2, len(phi_grid)), dtype=np.complex128)
        
        for n in range(dim):
            phi_wavefunc_amplitudes[0] += wavefunc_osc_basis_amplitudes[2 * n] * self.harm_osc_wavefunction(n, phi_basis_labels, l_osc)
            phi_wavefunc_amplitudes[1] += wavefunc_osc_basis_amplitudes[2 * n + 1] * self.harm_osc_wavefunction(n, phi_basis_labels, l_osc)

        return {
            "basis_labels": phi_basis_labels,
            "amplitudes": phi_wavefunc_amplitudes,
            "energy": evals[which]
        }
        
                
    def potential(self, phi: Union[float, np.ndarray]) -> np.ndarray:
        """
        Calculates the potential energy for given values of phi.

        Parameters
        ----------
        phi : Union[float, np.ndarray]
            The phase values at which to calculate the potential.

        Returns
        -------
        np.ndarray
            The potential energy values.
        """
        phi_array = np.atleast_1d(phi)
        evals_array = np.zeros((len(phi_array), 2))
        phi_ext = 2 * np.pi * self.phase

        for i, phi_val in enumerate(phi_array):
            if self.flux_grouping == 'ABS':
                inductive_term = 0.5 * self.El * phi_val**2 * np.eye(2)
                andreev_term = -self.Gamma * np.cos((phi_val + self.phase) / 2) * sigma_z() - self.delta_Gamma * np.sin((phi_val + self.phase) / 2) * sigma_y() + self.er * sigma_x()
            elif self.flux_grouping == 'L':
                inductive_term = 0.5 * self.El * (phi_val + phi_ext)**2 * np.eye(2)
                andreev_term = -self.Gamma * np.cos(phi_val / 2) * sigma_z() - self.delta_Gamma * np.sin(phi_val / 2) * sigma_y() + self.er * sigma_x()
            
            potential_operator = inductive_term + andreev_term
            evals_array[i] = eigh(
                potential_operator,
                eigvals_only=True,
                check_finite=False,
        )

        return evals_array
    
    def tphi_1_over_f(
        self, 
        A_noise: float, 
        i: int, 
        j: int, 
        noise_op: str,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        get_rate: bool = False,
        **kwargs
        ) -> float:
        """
        Calculates the 1/f dephasing time (or rate) due to an arbitrary noise source.

        Parameters
        ----------
        A_noise : float
            Noise strength.
        i : int
            State index that along with j defines a qubit.
        j : int
            State index that along with i defines a qubit.
        noise_op : str
            Name of the noise operator, typically Hamiltonian derivative w.r.t. noisy parameter.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors (default is None).
        get_rate : bool, optional
            Whether to return the rate instead of the Tphi time (default is False).

        Returns
        -------
        float
            The 1/f dephasing time (or rate).
        """
        p = {"omega_ir": 2 * np.pi * 1, "omega_uv": 3 * 2 * np.pi * 1e6, "t_exp": 10e-6}
        p.update(kwargs)
                
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(j, i) + 1)
        else:
            evals, evecs = esys

        noise_operator = getattr(self, noise_op)()    
        dEij_d_lambda = np.abs(evecs[i].conj().T @ noise_operator @ evecs[i] - evecs[j].conj().T @ noise_operator @ evecs[j])

        rate = (dEij_d_lambda * A_noise * np.sqrt(2 * np.abs(np.log(p["omega_ir"] * p["t_exp"]))))
        rate *= 2 * np.pi * 1e9 # Convert to rad/s

        return rate if get_rate else 1 / rate
    
    def tphi_1_over_f_flux(
        self, 
        A_noise: float = 1e-6,
        i: int = 0, 
        j: int = 1, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        get_rate: bool = False, 
        **kwargs
        ) -> float:
        return self.tphi_1_over_f(A_noise, i, j, 'd_hamiltonian_d_phase', esys=esys, get_rate=get_rate, **kwargs)

    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = 1,
        plot_potential: bool = False,
        basis: str = 'phase',
        rotate: bool = False,
        mode: str = 'abs',
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the wave function in the phi basis.

        Parameters
        ----------
        which : Union[int, Iterable[int]], optional
            Index or indices of desired wave function(s) (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors.
        scaling : float, optional
            Scaling factor for the wavefunction (default is 1).
        plot_potential : bool, optional
            Whether to plot the potential (default is False).
        basis: str, optional
            Basis in which to return the wavefunction ('phase' or 'charge') (default is 'phase').
        rotate : bool, optional
            Whether to rotate the basis (default is False).
        mode: str, optional
            Mode of the wavefunction ('abs', 'real', or 'imag') (default is 'abs').
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if isinstance(which, int):
            which = [which]
            
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)
            
        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle())
        else:
            fig, ax = fig_ax
        
        if plot_potential:
            potential = self.potential(phi=phi_grid)
            ax.plot(phi_grid, potential[:, 0], color='black', label='Potential')
            ax.plot(phi_grid, potential[:, 1], color='black')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys, basis=basis, rotate=rotate)
            phi_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]
            
            if mode == 'abs':
                y_values = np.abs(wavefunc_amplitudes[0])
                y_values_down = np.abs(wavefunc_amplitudes[1])
            elif mode == 'real':
                y_values = wavefunc_amplitudes[0].real
                y_values_down = wavefunc_amplitudes[1].real
            elif mode == 'imag':
                y_values = wavefunc_amplitudes[0].imag
                y_values_down = wavefunc_amplitudes[1].imag
            else:
                raise ValueError("Invalid mode; must be 'abs', 'real', or 'imag'.")

            ax.plot(
                phi_basis_labels,
                wavefunc_energy + scaling * y_values,
                label=rf"$\Psi_{idx} \uparrow $"
                )
            ax.plot(
                phi_basis_labels, 
                wavefunc_energy + scaling * y_values_down,
                label=rf"$\Psi_{idx} \downarrow $"
                )
            
        if basis == 'phase':
            ax.set_xlabel(r"$2 \pi \Phi / \Phi_0$")
            ax.set_ylabel(r"$\psi(\varphi)$, Energy [GHz]")
        elif basis == 'charge':
            ax.set_xlabel(r"$n$")
            ax.set_ylabel(r"$\psi(n)$, Energy [GHz]")
        
        ax.legend()
        ax.grid(True)

        return fig, ax      

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
#     'phase': phase_operator,
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
    # raise NotImplementedError
    # matrix_elements, eigenenergies = matrix_elements_vs_parameter(parameter_name, parameter_values, operator_name, fixed_params, state_i, state_j, plot=False)
    # t1 = hbar**2/np.abs(matrix_elements)**2/spectral_density(eigenenergies)
    # if plot:
    #     ylabel = f'T1'
    #     title = f'T1 vs {parameter_name}'
    #     plot_vs_parameters(parameter_values, t1, parameter_name, ylabel, title, filename)
    # return t1