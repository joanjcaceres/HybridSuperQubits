import numpy as np
from src.storage import SpectrumData
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm.notebook import tqdm
from scipy.special import factorial, pbdv
from qutip import Qobj, destroy, tensor, qeye, sigmaz, sigmay, sigmax
# import dynamiqs as dq

class Ferbo:
    """
    A class to represent a fermionic-bosonic qubit system.

    Attributes
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
    flux : float
        External magnetic flux.
    dimension : int
        Dimension of the Hilbert space.

    Methods
    -------
    charge_number_operator() -> Qobj:
        Returns the charge number operator.
    phase_operator() -> Qobj:
        Returns the phase operator.
    charge_number_operator_total() -> Qobj:
        Returns the total charge number operator.
    phase_operator_total() -> Qobj:
        Returns the total phase operator.
    jrl_potential() -> Qobj:
        Returns the Josephson Ring Ladder potential.
    hamiltonian() -> Qobj:
        Returns the Hamiltonian of the system.
    get_spectrum_vs_paramvals(param_name: str, param_vals: List[float], evals_count: int = 6, subtract_ground: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        Calculates the eigenenergies and eigenstates for a range of parameter values.
    matrixelement_table(operator: str, evecs: np.ndarray = None, evals_count: int = 6) -> np.ndarray:
        Returns a table of matrix elements for a given operator with respect to the eigenstates.
    get_matelements_vs_paramvals(operators: Union[str, List[str]], param_name: str, param_vals: np.ndarray, evals_count: int = 6) -> Dict[str, Dict[str, np.ndarray]]:
        Calculates the matrix elements for a list of operators over a range of parameter values.
    """
    def __init__(self, Ec, El, Gamma, delta_Gamma, er, flux, dimension):
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
        flux : float
            External magnetic flux.
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
        self.flux = flux
        self.dimension = dimension
        self.flux_grouping = flux_grouping
        
    def __repr__(self) -> str:
        """
        Returns a string representation of the Ferbo instance.

        Returns
        -------
        str
            A string representation of the Ferbo instance.
        """
        init_params = ['Ec', 'El', 'Gamma', 'delta_Gamma', 'er', 'flux', 'dimension', 'flux_grouping']
        init_dict = {name: getattr(self, name) for name in init_params}
        return f"{type(self).__name__}(**{init_dict!r})"
    
    def n_operator(self) -> Qobj:
        """
        Returns the charge number operator.

        Returns
        -------
        Qobj
            The charge number operator.
        """
        return 1j/2 * (self.El/2/self.Ec)**0.25 * (destroy(self.dimension).dag() - destroy(self.dimension))
    
    def phase_operator(self) -> Qobj:
        """
        Returns the phase operator.

        Returns
        -------
        Qobj
            The phase operator.
        """
        return (2*self.Ec/self.El)**0.25 * (destroy(self.dimension).dag() + destroy(self.dimension))
    
    def n_operator_total(self) -> Qobj:
        """
        Returns the total charge number operator.

        Returns
        -------
        Qobj
            The total charge number operator.
        """
        return tensor(self.n_operator(), qeye(2))
    
    def phase_operator_total(self) -> Qobj:
        """
        Returns the total phase operator.

        Returns
        -------
        Qobj
            The total phase operator.
        """
        return tensor(self.phase_operator(), qeye(2))
    
    def dH_d_flux(self) -> Qobj:
        """
        Returns the derivative of the Hamiltonian with respect to the external magnetic flux.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the external magnetic flux.
        """
        if self.flux_grouping == 'L':
            return - self.El * (self.phase_operator_total() + self.flux)
        elif self.flux_grouping == 'ABS':
            phase_op = self.phase_operator() - self.flux
            return self.Gamma/2 * tensor((phase_op/2).sinm(),sigmax()) - self.delta_Gamma/2 * tensor((phase_op/2).cosm(),sigmay())
                
    def dH_d_er(self) -> Qobj:
        """
        Returns the derivative of the Hamiltonian with respect to the energy relaxation rate.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the energy relaxation rate.
        """
        return - tensor(qeye(self.dimension),sigmaz())
    
    def dH_d_delta_Gamma(self) -> Qobj:
        """
        Returns the derivative of the Hamiltonian with respect to the coupling strength difference.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the coupling strength difference.
        """
        phase_op = self.phase_operator()
        return - tensor((phase_op/2).sinm(),sigmay())
    
    def jrl_potential(self) -> Qobj:
        """
        Returns the Josephson Ring Ladder potential.

        Returns
        -------
        Qobj
            The Josephson Ring Ladder potential.
        """
        phase_op = self.phase_operator()
        if self.flux_grouping == 'ABS':
            phase_op -= self.flux
        
        return  - self.Gamma * tensor((phase_op/2).cosm(),sigmax()) - self.delta_Gamma * tensor((phase_op/2).sinm(),sigmay()) - self.er * tensor(qeye(self.dimension),sigmaz())
    
    def hamiltonian(self) -> Qobj:
        """
        Returns the Hamiltonian of the system.

        Returns
        -------
        Qobj
            The Hamiltonian of the system.
        """
        charge_term = 4 * self.Ec * self.n_operator_total()**2
        if self.flux_grouping == 'ABS':
            inductive_term = 0.5 * self.El * self.phase_operator_total()**2
        else:
            inductive_term = 0.5 * self.El * (self.phase_operator_total() + self.flux)**2
            
        potential = self.jrl_potential()
        return charge_term + inductive_term + potential
    
    def phi_osc(self) -> float:
        """
        Returns the oscillator length for the LC oscillator composed of the inductance and capacitance.

        Returns
        -------
        float
            Oscillator length.
        """
        return (8.0 * self.Ec / self.El) ** 0.25
    
    def harm_osc_wavefunction(self, n: int, x: Union[float, np.ndarray], l_osc: float) -> Union[float, np.ndarray]:
        """
        Returns the value of the harmonic oscillator wave function.

        Parameters
        ----------
        n : int
            Index of wave function, n=0 is ground state.
        x : Union[float, np.ndarray]
            Coordinate(s) where wave function is evaluated.
        l_osc : float
            Oscillator length.

        Returns
        -------
        Union[float, np.ndarray]
            Value of harmonic oscillator wave function.
        """
        result = pbdv(n, np.sqrt(2.0) * x / l_osc) / np.sqrt(l_osc * np.sqrt(np.pi) * factorial(n))
        return result[0]
    
    def wavefunction(self, which: int = 0, phi_grid: np.ndarray = None, esys: Tuple[np.ndarray, np.ndarray] = None) -> Dict[str, Any]:
        """
        Returns a wave function in the phi basis.

        Parameters
        ----------
        which : int, optional
            Index of desired wave function (default is 0).
        phi_grid : np.ndarray, optional
            Custom grid for phi; if None, a default grid is used.

        Returns
        -------
        Dict[str, Any]
            Wave function data containing basis labels, amplitudes, and energy.
        """
        if esys is None:
            evals_count = max(which + 1, 3)
            H = self.hamiltonian()
            evals, evecs = H.eigenstates(eigvals=evals_count)
            evecs = np.array([evec.full().flatten() for evec in evecs])
        else:
            evals, evecs = esys
            
        dim = self.dimension
        
        # Change of basis
        U = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        U_qobj = Qobj(U)
        change_of_basis_operator = tensor(qeye(dim), U_qobj).full()
        evecs = (change_of_basis_operator @ evecs.T).T
                        
        if phi_grid is None:
            phi_grid = np.linspace(-4.5 * np.pi, 4.5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[which, :]
        phi_wavefunc_amplitudes = np.zeros((2, len(phi_grid)), dtype=np.complex_)
        phi_osc = self.phi_osc()
        for n in range(dim):
            phi_wavefunc_amplitudes[0] += wavefunc_osc_basis_amplitudes[2 * n] * self.harm_osc_wavefunction(n, phi_basis_labels, phi_osc)
            phi_wavefunc_amplitudes[1] += wavefunc_osc_basis_amplitudes[2 * n + 1] * self.harm_osc_wavefunction(n, phi_basis_labels, phi_osc)

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
        phi_ext = 2 * np.pi * self.flux

        for i, phi_val in enumerate(phi_array):
            if self.flux_grouping == 'ABS':
                inductive_term = 0.5 * self.El * phi_val**2 * qeye(2)
                andreev_term = -self.Gamma * np.cos((phi_val + self.flux) / 2) * sigmax() - self.delta_Gamma * np.sin((phi_val + self.flux) / 2) * sigmay() - self.er * sigmaz()
            elif self.flux_grouping == 'L':
                inductive_term = 0.5 * self.El * (phi_val + phi_ext)**2 * qeye(2)
                andreev_term = -self.Gamma * np.cos(phi_val / 2) * sigmax() - self.delta_Gamma * np.sin(phi_val / 2) * sigmay() - self.er * sigmaz()
            
            potential_operator = inductive_term + andreev_term
            evals_array[i] = potential_operator.eigenenergies()

        return evals_array

    
    def get_spectrum_vs_paramvals(self, param_name: str, param_vals: List[float], evals_count: int = 6, subtract_ground: bool = False)  -> SpectrumData:
        """
        Calculates the eigenenergies and eigenstates for a range of parameter values.

        Parameters
        ----------
        param_name : str
            The name of the parameter to vary.
        param_vals : List[float]
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        subtract_ground : bool, optional
            Whether to subtract the ground state energy from the eigenenergies (default is False).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The eigenenergies and eigenstates for the range of parameter values.
        """
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
        
        return SpectrumData(
            energy_table=np.array(eigenenergies_array),
            system_params=self.__dict__,
            param_name=param_name,
            param_vals=np.array(param_vals),
            state_table=np.array(eigenstates_array)
        )    
        
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


    def get_matelements_vs_paramvals(self, operators: Union[str, List[str]], param_name: str, param_vals: np.ndarray, evals_count: int = 6) -> SpectrumData:
        """
        Calculates the matrix elements for a list of operators over a range of parameter values.

        Parameters
        ----------
        operators : Union[str, List[str]]
            The name(s) of the operator(s).
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            The matrix elements for the operators over the range of parameter values.
        """
        if isinstance(operators, str):
            operators = [operators]
                        
        paramvals_count = len(param_vals)
            
        eigenenergies_array = np.empty((paramvals_count, evals_count))
        eigenstates_array = np.empty((paramvals_count, evals_count, 2*self.dimension), dtype=np.complex_)
        matrixelem_tables = {operator: np.empty((paramvals_count, evals_count, evals_count), dtype=complex) for operator in operators}
            
        for idx, val in enumerate(tqdm(param_vals)):
            setattr(self, param_name, val)
            H = self.hamiltonian()
            eigenenergies, eigenstates = H.eigenstates(eigvals=evals_count)
            eigenenergies_array[idx] = eigenenergies
            eigenstates_array[idx] = np.array([eigenstate.full().flatten() for eigenstate in eigenstates])
            
            for operator in operators:
                matrix_elements = self.matrixelement_table(operator, evecs=eigenstates, evals_count=evals_count)
                matrixelem_tables[operator][idx] = matrix_elements
                
        spectrum_data = SpectrumData(
            energy_table=eigenenergies_array,
            system_params=self.__dict__,
            param_name=param_name,
            param_vals=param_vals,
            state_table=eigenstates_array,
            matrixelem_table=matrixelem_tables
        )
        
        return spectrum_data
    
    def get_t1_vs_paramvals(self, noise_channels: Union[str, List[str]], param_name: str, param_vals: np.ndarray, i: int = 1, j: int = 0, spectrum_data: SpectrumData = None, **kwargs) -> SpectrumData:
        """
        Calculates the T1 times for given noise channels over a range of parameter values.

        Parameters
        ----------
        noise_channels : Union[str, List[str]]
            The noise channels to calculate ('capacitive', 'inductive', etc.).
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        i : int, optional
            The initial state index (default is 1).
        j : int, optional
            The final state index (default is 0).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the T1 calculation method.

        Returns
        -------
        SpectrumData
            The T1 times for the specified noise channels over the range of parameter values.
        """
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]
            
        evals_count=max(i, j) + 1
        
        if spectrum_data is None:
            spectrum_data = self.get_matelements_vs_paramvals(noise_channels, param_name, param_vals, evals_count=evals_count)
        else:
            operators = []
            if 'capacitive' in noise_channels and 'n_operator_total' not in spectrum_data.matrixelem_table:
                operators.append('n_operator_total')
            if 'inductive' in noise_channels and 'phase_operator_total' not in spectrum_data.matrixelem_table:
                operators.append('phase_operator_total')
            if operators:
                if spectrum_data.state_table is not None:
                    for operator in operators:
                        matrixelem_table = np.empty((len(param_vals), evals_count, evals_count), dtype=np.complex_)
                        for index, paramval in enumerate(param_vals):
                            evecs = spectrum_data.state_table[index]
                            matrixelem_table[index] = self.matrixelement_table(operator, evecs=evecs, evals_count=evals_count)
                        spectrum_data.matrixelem_table[operator] = matrixelem_table
                else:
                    spectrum_data = self.get_matelements_vs_paramvals(operators, param_name, param_vals, evals_count=evals_count)
    
        paramvals_count = len(param_vals)
        t1_tables = {channel: np.empty(paramvals_count, dtype=np.float_) for channel in noise_channels}

        for index, paramval in enumerate(param_vals):
            esys = (spectrum_data.energy_table[index], spectrum_data.state_table[index])
            for channel in noise_channels:
                if channel == 'capacitive':
                    t1_tables[channel][index] = self.t1_capacitive(i=i, j=j, esys=esys, matrix_elements=spectrum_data.matrixelem_table['n_operator_total'][index], **kwargs)
                elif channel == 'inductive':
                    t1_tables[channel][index] = self.t1_inductive(i=i, j=j, esys=esys, matrix_elements=spectrum_data.matrixelem_table['phase_operator_total'][index], **kwargs)
                else:
                    raise ValueError(f"Unsupported T1 noise channel: {channel}")

        spectrum_data.t1_table = t1_tables
        return spectrum_data
    
    def get_tphi_vs_paramvals(
        self, 
        noise_channels: Union[str, List[str]],
        param_name: str,
        param_vals: np.ndarray,
        i: int = 1, 
        j: int = 0,
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> SpectrumData:
        """
        Calculates the Tphi times for given noise channels over a range of parameter values.

        Parameters
        ----------
        noise_channels : Union[str, List[str]]
            The noise channels to calculate ('flux', etc.).
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        i : int, optional
            The initial state index (default is 1).
        j : int, optional
            The final state index (default is 0).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the Tphi calculation method.

        Returns
        -------
        SpectrumData
            The Tphi times for the specified noise channels over the range of parameter values.
        """
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]
            
        evals_count = max(i, j) + 1
        
        if spectrum_data is None:
            spectrum_data = self.get_matelements_vs_paramvals(noise_channels, param_name, param_vals, evals_count=evals_count)
        else:
            operators = []
            if 'flux' in noise_channels and 'dH_d_flux' not in spectrum_data.matrixelem_table:
                operators.append('dH_d_flux')
            if operators:
                if spectrum_data.state_table is not None:
                    for operator in operators:
                        matrixelem_table = np.empty((len(param_vals), evals_count, evals_count), dtype=np.complex_)
                        for index, paramval in enumerate(param_vals):
                            evecs = spectrum_data.state_table[index]
                            matrixelem_table[index] = self.matrixelement_table(operator, evecs=evecs, evals_count=evals_count)
                        spectrum_data.matrixelem_table[operator] = matrixelem_table
                else:
                    spectrum_data = self.get_matelements_vs_paramvals(operators, param_name, param_vals, evals_count=evals_count)

        paramvals_count = len(param_vals)
        tphi_tables = {(i, j, channel): np.empty(paramvals_count, dtype=np.float_) for channel in noise_channels}

        for index, paramval in enumerate(param_vals):
            esys = (spectrum_data.energy_table[index], spectrum_data.state_table[index])
            for channel in noise_channels:
                if channel == 'flux':
                    A_noise = kwargs.pop('A_noise', 1e-6)
                    tphi_tables[(i, j, channel)][index] = self.tphi_1_over_f_flux(A_noise=A_noise, i=i, j=j, esys=esys, **kwargs)
                else:
                    raise ValueError(f"Unsupported Tphi noise channel: {channel}")

        spectrum_data.tphi_table = tphi_tables
        return spectrum_data
        
        if Q_cap is None:
            Q_cap_fun = lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(omega))**0.7
        elif callable(Q_cap):
            Q_cap_fun = Q_cap
        else:
            Q_cap_fun = lambda omega: Q_cap

        def spectral_density(omega, T):
            return 32 * np.pi * (self.Ec * 1e9) / Q_cap_fun(omega) * 1/np.tanh(hbar * np.abs(omega) / (2 * k * T))

        noise_op = noise_op or self.n_operator_total()
        if isinstance(noise_op, Qobj):
            noise_op = noise_op.full()
            
        if esys is None:
            H = self.hamiltonian()
            evals, evecs = H.eigenstates(eigvals=max(i, j) + 1)
        else:
            evals, evecs = esys
            
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        
        s = spectral_density(omega, T)
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table('n_operator_total', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = matrix_element**2 * s
        return rate if get_rate else 1 / rate
    
    def t1_inductive(self, i: int = 1, j: int = 0, Q_ind: float = 500e6, T: float = 0.015, esys: Tuple[np.ndarray, np.ndarray] = None, matrix_elements: np.ndarray = None, get_rate: bool = False, noise_op: Optional[Union[np.ndarray, Qobj]] = None) -> float:
        
        def spectral_density(omega, T):
            return 4 * np.pi * (self.El * 1e9) / Q_ind * 1 / np.tanh(hbar * np.abs(omega) / (2 * k * T))

        noise_op = noise_op or self.phase_operator_total()
        if isinstance(noise_op, Qobj):
            noise_op = noise_op.full()
            
        if esys is None:
            H = self.hamiltonian()
            evals, evecs = H.eigenstates(eigvals=max(i, j) + 1)
        else:
            evals, evecs = esys
            
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        s = spectral_density(omega, T)
        
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table('phase_operator_total', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = matrix_element**2 * s
        return rate if get_rate else 1 / rate
    
    def tphi_1_over_f(
        self, 
        A_noise: float, 
        i: int, 
        j: int, 
        noise_op: Union[np.ndarray, Qobj],
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
        noise_op : Union[np.ndarray, Qobj]
            Noise operator, typically Hamiltonian derivative w.r.t. noisy parameter.
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
            H = self.hamiltonian()
            evals, evecs = H.eigenstates(eigvals=max(j, i) + 1)
            evecs = np.array([evec.full().flatten() for evec in evecs])
        else:
            evals, evecs = esys

        if isinstance(noise_op, np.ndarray):
            dEij_d_lambda = np.abs(evecs[i, :].conj().T @ noise_op @ evecs[i, :] - evecs[j, :].conj().T @ noise_op @ evecs[j, :])
        elif isinstance(noise_op, Qobj):
            dEij_d_lambda = np.abs(evecs[i, :].conj().T @ noise_op.full() @ evecs[i, :] - evecs[j, :].conj().T @ noise_op.full() @ evecs[j, :])
        else:
            raise ValueError("Noise operator must be a numpy array or Qobj.")

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
        return self.tphi_1_over_f(A_noise, i, j, self.dH_d_flux(), esys=esys, get_rate=get_rate, **kwargs)

    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = None,
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
            
        potential = self.potential(phi=phi_grid)

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(rf'$E_c = {self.Ec}, E_l = {self.El}, \Gamma = {self.Gamma}, \delta \Gamma = {self.delta_Gamma}, \epsilon_r = {self.er}$')
        else:
            fig, ax = fig_ax
        
        ax.plot(phi_grid/2/np.pi, potential[:, 0], color='black', label='Potential')
        ax.plot(phi_grid/2/np.pi, potential[:, 1], color='black')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys)
            phi_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]

            ax.plot(
                phi_basis_labels/2/np.pi,
                wavefunc_energy + scaling * (wavefunc_amplitudes[0].real + wavefunc_amplitudes[0].imag),
                # color="blue",
                label=rf"$\Psi_{idx} \uparrow $"
                )
            ax.plot(
                phi_basis_labels/2/np.pi, 
                wavefunc_energy + scaling * (wavefunc_amplitudes[1].real + wavefunc_amplitudes[1].imag),
                # color="red",
                label=rf"$\Psi_{idx} \downarrow $"
                )

        ax.set_xlabel(r"$\Phi / \Phi_0$")
        ax.set_ylabel(r"$\psi(\varphi)$, Energy [GHz]")
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
    # raise NotImplementedError
    # matrix_elements, eigenenergies = matrix_elements_vs_parameter(parameter_name, parameter_values, operator_name, fixed_params, state_i, state_j, plot=False)
    # t1 = hbar**2/np.abs(matrix_elements)**2/spectral_density(eigenenergies)
    # if plot:
    #     ylabel = f'T1'
    #     title = f'T1 vs {parameter_name}'
    #     plot_vs_parameters(parameter_values, t1, parameter_name, ylabel, title, filename)
    # return t1