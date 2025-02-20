import numpy as np
import matplotlib.pyplot as plt
import scqubits.utils.plotting as plot
from abc import ABC, abstractmethod
from scipy.linalg import expm, eigh
from scipy.special import factorial, pbdv
from .storage import SpectrumData
from tqdm.notebook import tqdm
from scipy.constants import hbar, k, h, e
from scipy.special import k0
from typing import Tuple, List, Union, Any, Callable, Optional

class QubitBase(ABC):
    PARAM_LABELS = {}
    OPERATOR_LABELS = {}
    
    def __init__(self, dimension:int):
        self.dimension = dimension
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the QubitBase instance.

        Returns
        -------
        str
            A string representation of the QubitBase instance.
        """
        init_params = [param for param in self.__dict__ if not param.startswith('_')]
        init_dict = {name: getattr(self, name) for name in init_params}
        return f"{type(self).__name__}(**{init_dict!r})"
    
    @abstractmethod
    def n_operator(self) -> np.ndarray:
        """
        Returns the number operator for the qubit.

        Returns
        -------
        ndarray
            The number operator for the qubit.
        """
        pass
    
    def displacement_operator(self) -> np.ndarray:
        """
        Returns the displacement operator.

        Returns
        -------
        np.ndarray
            The displacement operator.
        """
        return expm(-1j*2*np.pi*self.n_operator())
    
    @abstractmethod
    def hamiltonian(self) -> np.ndarray:
        """
        Returns the Hamiltonian for the qubit.

        Returns
        -------
        ndarray
            The Hamiltonian for the qubit.
        """
        pass
    
    def eigensys(self, evals_count: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates eigenvalues and corresponding eigenvectors using scipy.linalg.eigh.

        Parameters
        ----------
        evals_count : int, optional
            Number of desired eigenvalues/eigenstates (default is 6).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Eigenvalues and eigenvectors as numpy arrays.
        """
        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigh(
            hamiltonian_mat,
            eigvals_only=False,
            subset_by_index=(0, evals_count - 1),
            # check_finite=False,
        )
        return evals, evecs
    
    def eigenvals(self, evals_count: int = 6) -> np.ndarray:
        """
        Calculates eigenvalues using scipy.linalg.eigh.

        Parameters
        ----------
        evals_count : int, optional
            Number of desired eigenvalues (default is 6).

        Returns
        -------
        np.ndarray
            Eigenvalues as a numpy array.
        """
        hamiltonian_mat = self.hamiltonian()
        evals = eigh(
            hamiltonian_mat,
            eigvals_only=True,
            subset_by_index=(0, evals_count - 1),
            check_finite=False,
        )
        return np.sort(evals)
    
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
        
        initial_value = getattr(self, param_name)
        
        for val in tqdm(param_vals, leave=False):
            self.set_param(param_name, val)
            eigenenergies, eigenstates = self.eigensys(evals_count)
            eigenenergies_array.append(eigenenergies)
            eigenstates_array.append(eigenstates)
            
        self.set_param(param_name, initial_value)
            
        spectrum_data = SpectrumData(
            energy_table=np.array(eigenenergies_array),
            system_params=self.__dict__,
            param_name=param_name,
            param_vals=np.array(param_vals),
            state_table=np.array(eigenstates_array)
        )

        if subtract_ground:
            spectrum_data.subtract_ground()
        
        return spectrum_data  
        
    def matrixelement_table(self, operator: str, evecs: np.ndarray = None, evals_count: int = 6) -> np.ndarray:
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
            _, evecs = self.eigensys(evals_count)
            
        operator_matrix = getattr(self, operator)()
        matrix_elements = evecs.conj().T @ operator_matrix @ evecs
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
        eigenstates_array = np.empty((paramvals_count, self.dimension, evals_count), dtype=np.complex128)
        matrixelem_tables = {operator: np.empty((paramvals_count, evals_count, evals_count), dtype=complex) for operator in operators}
        
        initial_value = getattr(self, param_name)
            
        for idx, val in enumerate(tqdm(param_vals, leave=False)):
            self.set_param(param_name, val)
            eigenenergies, eigenstates = self.eigensys(evals_count)
            eigenenergies_array[idx] = eigenenergies
            eigenstates_array[idx] = eigenstates
            
            for operator in operators:
                matrix_elements = self.matrixelement_table(operator, evecs=eigenstates, evals_count=evals_count)
                matrixelem_tables[operator][idx] = matrix_elements
                
        self.set_param(param_name, initial_value)
                            
        spectrum_data = SpectrumData(
            energy_table=eigenenergies_array,
            system_params=self.__dict__,
            param_name=param_name,
            param_vals=param_vals,
            state_table=eigenstates_array,
            matrixelem_table=matrixelem_tables
        )
        
        return spectrum_data
    
    def harm_osc_wavefunction(self, n: int, x: Union[float, np.ndarray], l_osc: float) -> Union[float, np.ndarray]:
        """
        Returns the value of the harmonic oscillator wave function.
        
        The wave function is computed using the parabolic cylinder function Dν(z), 
        which satisfies the Weber differential equation. This implementation 
        is based on the connection between the wave function of the harmonic 
        oscillator and the solutions to the Weber equation.

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

        References
        ----------
        - ParabolicCylinderD[ν, z]: https://reference.wolfram.com/language/ref/ParabolicCylinderD.html
        - The wave functions of the quantum harmonic oscillator are proportional 
        to the parabolic cylinder functions Dν(z).

        """
        result = pbdv(n, x / l_osc) / np.sqrt(np.sqrt(2 * np.pi) * factorial(n))
        return result[0]
    
    def plot_matrixelements(
        self,
        operator: str,
        evecs: np.ndarray = None,
        evals_count: int = 6,
        mode: str = "abs",
        show_numbers: bool = False,
        show_colorbar: bool = True,
        show3d: bool = True,
        **kwargs,
    ) -> Union[Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]], Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the matrix elements for a given operator with respect to the eigenstates.

        Parameters
        ----------
        operator : str
            The name of the operator.
        evecs : np.ndarray, optional
            The eigenstates (default is None, in which case they are calculated).
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        mode : str, optional
            The mode for displaying matrix elements ('abs', 'real', 'imag') (default is 'abs').
        show_numbers : bool, optional
            Whether to show the matrix element values as numbers (default is False).
        show_colorbar : bool, optional
            Whether to show a colorbar (default is True).
        show3d : bool, optional
            Whether to show a 3D plot (default is True).

        Returns
        -------
        Union[Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]], Tuple[plt.Figure, plt.Axes]]
            The figure and axes of the plot.
        """
        # Obtener la tabla de elementos de la matriz utilizando el método matrixelement_table
        matrix_elements = self.matrixelement_table(operator, evecs, evals_count)
        
        modefunction = {
            "abs": np.abs,
            "real": np.real,
            "imag": np.imag
        }.get(mode, None)
        
        if modefunction is None:
            raise ValueError(f"Unsupported mode: {mode}")

        matrix_elements = modefunction(matrix_elements)

        if show3d:
            return plot.matrix(
                matrix_elements,
                mode=mode,
                show_numbers=show_numbers,
                **kwargs,
            )

        return plot.matrix2d(
            matrix_elements,
            mode=mode,
            show_numbers=show_numbers,
            show_colorbar=show_colorbar,
            **kwargs,
        )
        
    def t1_capacitive(
        self,
        i: int = 1, 
        j: int = 0, 
        Q_cap: Union[float, Callable] = None,
        T: float = 0.015, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        matrix_elements: np.ndarray = None, 
        get_rate: bool = False,
        noise_op: Optional[np.ndarray] = None
        ) -> float:
        
        if Q_cap is None:
            Q_cap_fun = lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(omega))**0.7 # Assuming that Ec is in GHz
        elif callable(Q_cap):
            Q_cap_fun = Q_cap
        else:
            Q_cap_fun = lambda omega: Q_cap

        def spectral_density(omega, T):
            # Assuming that Ec is in GHz
            return 8 * self.Ec / Q_cap_fun(omega) * 1/np.tanh(hbar * np.abs(omega) / (2 * k * T))

        noise_op = noise_op or self.n_operator()
            
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
            
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        
        s = spectral_density(omega, T)
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table('n_operator', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = 2 * np.pi * np.abs(matrix_element)**2 * s
        rate *= 1e9
        
        return rate if get_rate else 1 / rate
    
    def t1_inductive(
        self, 
        i: int = 1, 
        j: int = 0,
        Q_ind: float = None,
        T: float = 0.015, 
        esys: Tuple[np.ndarray, np.ndarray] = None,
        matrix_elements: np.ndarray = None, 
        get_rate: bool = False,
        ) -> float:
        
        if Q_ind is None:
            Q_ind_fun = lambda omega: 500e6 * (
                k0(h * 0.5e9 / (2 * k * T)) * np.sinh(h * 0.5e9 / (2 * k * T))
                / (
                    k0(hbar * np.abs(omega) / (2 * k * T))
                    * np.sinh(hbar * np.abs(omega) / (2 * k * T))
                )
            )
        elif callable(Q_ind):
            Q_ind_fun = Q_ind
        else:
            Q_ind_fun = lambda omega: Q_ind
        
        def spectral_density(omega, T):
            return 2 * self.El / Q_ind_fun(omega) * 1 / np.tanh(hbar * np.abs(omega) / (2 * k * T))
            
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
            
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        s = spectral_density(omega, T)
        
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table('phase_operator', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = 2 * np.pi * matrix_element**2 * s
        rate *= 1e9  # Convert to GHz
        return rate if get_rate else 1 / rate
    
    def t1_flux_bias_line(
        self, 
        i: int = 1, 
        j: int = 0,
        M: float = 2500,
        Z: float = 50,
        T: float = 0.015, 
        esys: Tuple[np.ndarray, np.ndarray] = None,
        matrix_elements: np.ndarray = None, 
        get_rate: bool = False,
        ) -> float:
        
        def spectral_density(omega, T):
            return 4 * np.pi**2 * M**2 * np.abs(omega) * 1e9 * h / Z * (1 + 1/np.tanh(hbar * np.abs(omega) / (2 * k * T)))
            
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
            
        omega = 2 * np.pi * (evals[i] - evals[j]) * 1e9  # Convert to rad/s
        s = spectral_density(omega, T)
        
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table('d_hamiltonian_d_phase', evecs=evecs, evals_count=max(i, j) + 1)
        matrix_element = np.abs(matrix_elements[i, j])

        rate = 2 * np.pi * matrix_element**2 * s
        rate *= 1e9  # Convert to GHz
        return rate if get_rate else 1 / rate
        
    def get_t1_vs_paramvals(
        self, 
        noise_channels: Union[str, List[str]],
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> SpectrumData:
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
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
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
            
            if spectrum_data is not None:
                param_name = spectrum_data.param_name
                param_vals = spectrum_data.param_vals
                evals_count = spectrum_data.energy_table.shape[1]
            elif param_name is None or param_vals is None:
                raise ValueError("If spectrum_data is None, param_name and param_vals must be provided.")
                        
        if 'capacitive' in noise_channels:
            spectrum_data = self.get_t1_capacitive_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'inductive' in noise_channels:
            spectrum_data = self.get_t1_inductive_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'charge_impedance' in noise_channels:
            spectrum_data = self.get_t1_charge_impedance_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'critical_current' in noise_channels:
            spectrum_data = self.get_t1_critical_current_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'flux_bias_line' in noise_channels:
            spectrum_data = self.get_t1_flux_bias_line_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'flux_noise' in noise_channels:
            spectrum_data = self.get_t1_1_over_f_flux_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
        if 'Andreev' in noise_channels:
            spectrum_data = self.get_t1_er_vs_paramvals(param_name, param_vals, evals_count, spectrum_data, **kwargs)
            
        return spectrum_data

    def get_t1_capacitive_vs_paramvals(
        self, 
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        evals_count: int = 6, 
        spectrum_data: SpectrumData = None,
        Q_cap: Union[float, Callable] = None,
        T: float = 0.015, 
        **kwargs
        ) -> SpectrumData:
        """
        Calculates the T1 times for capacitive noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        Q_cap : Union[float, Callable], optional
            The capacitance quality factor or a function that returns it (default is None).
        T : float, optional
            The temperature (default is 0.015).
        **kwargs
            Additional arguments to pass to the T1 calculation method.

        Returns
        -------
        SpectrumData
            The T1 times for capacitive noise over the range of parameter values.
        """
        if Q_cap is None:
            Q_cap_fun = lambda omega: 1e6 * (2 * np.pi * 6e9 / np.abs(omega))**0.7 # Assuming that Ec is in GHz
        elif callable(Q_cap):
            Q_cap_fun = Q_cap
        else:
            Q_cap_fun = lambda omega: Q_cap

        def spectral_density(omega, T):
            # Assuming that Ec is in GHz
            return 8 * self.Ec / Q_cap_fun(omega) * 1/np.tanh(hbar * np.abs(omega) / (2 * k * T))

        noise_operator = 'n_operator'
        noise_channel = 'capacitive'
        
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel, T, **kwargs
        )
    
    def get_t1_inductive_vs_paramvals(
        self, 
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        evals_count: int = 6, 
        spectrum_data: SpectrumData = None,
        Q_ind: float = None,
        T: float = 0.015, 
        **kwargs
        ) -> SpectrumData:
        """
        Calculates the T1 times for inductive noise over a range of parameter values.

        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        Q_ind : float, optional
            The inductance quality factor (default is 500e6).
        T : float, optional
            The temperature (default is 0.015).
        **kwargs
            Additional arguments to pass to the T1 calculation method.

        Returns
        -------
        SpectrumData
            The T1 times for inductive noise over the range of parameter values.
        """
        if Q_ind is None:
            Q_ind_fun = lambda omega: 500e6 * (
                k0(h * 0.5e9 / (2 * k * T)) * np.sinh(h * 0.5e9 / (2 * k * T))
                / (
                    k0(hbar * np.abs(omega) / (2 * k * T))
                    * np.sinh(hbar * np.abs(omega) / (2 * k * T))
                )
            )
        elif callable(Q_ind):
            Q_ind_fun = Q_ind
        else:
            Q_ind_fun = lambda omega: Q_ind
        def spectral_density(omega, T):
            return 2 * self.El / Q_ind_fun(omega) * 1 / np.tanh(hbar * np.abs(omega) / (2 * k * T))

        noise_operator = 'phase_operator'
        noise_channel = 'inductive'
        
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel, T, **kwargs
        )
        
    def get_t1_charge_impedance_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        Z: float = 50,
        T: float = 0.015,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the T1 times for charge impedance noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        Z : float, optional
            The impedance (default is 50).
        T : float, optional
            The temperature (default is 0.015).
        **kwargs
        
        Returns
        -------
        SpectrumData
            The T1 times for charge impedance noise over the range of parameter values.
        """
        def spectral_density(omega, T):
            Rk = h / ((2 * e)**2)
            return omega/1e9 / Rk * Z * (1 + 1/np.tanh(hbar * np.abs(omega) / (2 * k * T)))

        noise_operator = 'n_operator'
        noise_channel = 'charge_impedance'
        
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel, T, **kwargs
        )
        
    def get_t1_flux_bias_line_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        M: float = 2500,
        Z: float = 50,
        T: float = 0.015,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the T1 times for flux bias line noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        M : float, optional
            The mutual inductance in units of Phi_0 = h/(2e) (default is 26000).
        Z : float, optional
            The impedance (default is 50).
        T : float, optional
            The temperature (default is 0.015).
        **kwargs
            Additional arguments to pass to the T1 calculation method.
            
        Returns
        -------
        SpectrumData
            The T1 times for flux bias line noise over the range of parameter values.
        """
        
        def spectral_density(omega, T):
            return 4 * np.pi**2 * M**2 * np.abs(omega) * 1e9 * h / Z * (1 + 1/np.tanh(hbar * np.abs(omega) / (2 * k * T)))
        
        noise_operator = 'd_hamiltonian_d_phase'
        noise_channel = 'flux_bias_line'
        
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel, T, **kwargs
        )
        
    def get_t1_1_over_f_flux_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        A_noise: float = 1e-6,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the T1 times for 1/f flux noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        A_noise : float, optional
            The amplitude of the noise (default is 1e-6).
        **kwargs
            Additional arguments to pass to the T1 calculation method.
            
        Returns
        -------
        SpectrumData
            The T1 times for 1/f flux noise over the range of parameter values.
        """
        def spectral_density(omega, T):
            return 2 * np.pi * A_noise**2 / np.abs(omega)
        
        noise_operator = 'd_hamiltonian_d_phase'
        noise_channel = 'flux_noise'
        
        T = 0.015
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel,T, **kwargs
        )

    def get_t1_critical_current_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        A_noise: float = 1e-7,
        N: int = 100,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the T1 times for critical current noise over a range of parameter values.

        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        A_noise : float, optional
            The amplitude of the noise (default is 1e-7).
        N : int, optional
            The number of junctions (default is 100).

        Returns
        -------
        SpectrumData
            The T1 times for critical current noise over the range of parameter values.
        """
        
        def spectral_density(omega, T):
            return 2 * np.pi * (A_noise * self.El / np.sqrt(N))**2 / omega * 1e9
        
        noise_operator = 'd_hamiltonian_d_EL'
        noise_channel = 'critical_current'
        
        T = 0.015
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel,T, **kwargs
        )
        
    def get_t1_er_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        A_noise: float = 0.04,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the T1 times for the Fermi level noise over a range of parameter values.

        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        A_noise : float, optional
            The amplitude of the noise (default is 0.04 GHz).


        Returns
        -------
        SpectrumData
            The T1 times for Fermi level noise over the range of parameter values.
        """
        
        def spectral_density(omega, T):
            return 2 * np.pi * 1e9 * A_noise**2 *np.abs(1/(omega*1e-9) + 0.01 * omega*1e-9 * 1/np.tanh(hbar * omega / (2 * k * T)))
        
        noise_operator = 'd_hamiltonian_d_er'
        noise_channel = 'Andreev'
        
        T = 0.015
        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data, spectral_density, noise_operator, noise_channel,T, **kwargs
        )
        
        
    def _get_t1_vs_paramvals(
        self, 
        param_name: str, 
        param_vals: np.ndarray, 
        evals_count: int, 
        spectrum_data: SpectrumData, 
        spectral_density: Callable, 
        noise_operator: str, 
        noise_channel: str,
        T: float, 
        **kwargs
        ) -> SpectrumData:
        """
        General method to calculate T1 times for a given noise channel over a range of parameter values.
        
        Parameters
        ----------
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        evals_count : int
            The number of eigenvalues and eigenstates to calculate.
        spectrum_data : SpectrumData
            Precomputed spectral data to use.
        spectral_density : Callable
            Function to calculate the spectral density.
        noise_operator : str
            The noise operator to use ('n_operator' or 'phase_operator').
        noise_channel : str
            The noise channel to use ('capacitive' or 'inductive').
        T : float
            The temperature.
        **kwargs
            Additional arguments to pass to the T1 calculation method.

        Returns
        -------
        SpectrumData
            The T1 times for the specified noise channel over the range of parameter values.
        """
        if spectrum_data is None or noise_operator not in spectrum_data.matrixelem_table:
            spectrum_data = self.get_matelements_vs_paramvals(noise_operator, param_name, param_vals, evals_count=evals_count)
            
        evals_array = spectrum_data.energy_table
        transition_table = evals_array[:, :, np.newaxis] - evals_array[:, np.newaxis, :]
        
        omega = 2 * np.pi * transition_table * 1e9
        epsilon = 1e-12  # Pequeña constante para evitar divisiones por cero
        omega = np.where(omega == 0, epsilon, omega) 
        
        s = spectral_density(omega, T)
        
        matrix_element = spectrum_data.matrixelem_table[noise_operator]
        rate = 2 * np.pi * np.abs(matrix_element)**2 * s
        
        rate *= 1e9  # Convert to rad/s
        rate = np.where(rate == 0, epsilon, rate)
        t1_table = 1 / rate
        
        for idx in range(t1_table.shape[0]):
            np.fill_diagonal(t1_table[idx], np.nan)
        
        spectrum_data.t1_table[noise_channel] = t1_table        
        return spectrum_data
    
    def _get_tphi_vs_paramvals(
        self, 
        param_name: str, 
        param_vals: np.ndarray, 
        A_noise: float,
        noise_channel: str,
        noise_operator: str,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> SpectrumData:
        """
        Calculates the Tphi times for given noise channels over a range of parameter values.

        Parameters
        ----------
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        A_noise : float
            The amplitude of the noise.
        noise_channel : str
            The noise channel to calculate ('flux', etc.).
        noise_operator : str
            The noise operator to use.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the Tphi calculation method.

        Returns
        -------
        SpectrumData
            The Tphi times for the specified noise channels over the range of parameter values.
        """
        p = {"omega_ir": 2 * np.pi * 1, "omega_uv": 3 * 2 * np.pi * 1e6, "t_exp": 10e-6}
        p.update(kwargs)
            
        if spectrum_data is None or noise_operator not in spectrum_data.matrixelem_table:
            new_spec = self.get_matelements_vs_paramvals(noise_operator, param_name, param_vals, evals_count=evals_count)
            spectrum_data.matrixelem_table[noise_operator] = new_spec.matrixelem_table[noise_operator]
            
        param_vals = spectrum_data.param_vals
                            
        dE_d_lambda = np.diagonal(spectrum_data.matrixelem_table[noise_operator], axis1=1, axis2 = 2)
        dEij_d_lambda = dE_d_lambda[:,:,np.newaxis] - dE_d_lambda[:,np.newaxis,:]
        
        if spectrum_data.param_name == "phase":
            d2Eij_d_lambda2 = np.gradient(dEij_d_lambda, param_vals, axis=0, edge_order=2)
            rate_1er = np.abs(dEij_d_lambda) * A_noise * np.sqrt(2 * np.abs(np.log(p["omega_ir"] * p["t_exp"])))
            rate_2nd = np.abs(d2Eij_d_lambda2) * A_noise**2  * np.sqrt(2 * np.log(p['omega_uv']/p['omega_ir'])**2 + 2 * np.log(p["omega_ir"] * p["t_exp"])**2)
            
            rate = np.sqrt(rate_1er**2 + rate_2nd**2)
        else:
            rate = np.abs(dEij_d_lambda) * A_noise * np.sqrt(2 * np.abs(np.log(p["omega_ir"] * p["t_exp"])))
        
        epsilon = 1e-12  # Pequeña constante para evitar divisiones por cero
        rate = np.where(rate == 0, epsilon, rate) 
        rate *= 2 * np.pi * 1e9 # Convert to rad/s
        tphi_table = 1 / rate
        
        for idx in range(tphi_table.shape[0]):
            np.fill_diagonal(tphi_table[idx], np.nan)
            
        spectrum_data.tphi_table[noise_channel] = tphi_table
        
        return spectrum_data

    def get_tphi_flux_vs_paramvals(
        self, 
        param_name: str, 
        param_vals: np.ndarray, 
        A_noise: float = 1e-6,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> SpectrumData:
        """
        Calculates the Tphi times for flux noise over a range of parameter values.

        Parameters
        ----------
        param_name : str
            The name of the parameter to vary.
        param_vals : np.ndarray
            The values of the parameter to vary.
        A_noise : float
            The amplitude of the noise.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the Tphi calculation method.

        Returns
        -------
        SpectrumData
            The Tphi times for flux noise over the range of parameter values.
        """
        return self._get_tphi_vs_paramvals(
            param_name=param_name, 
            param_vals=param_vals, 
            A_noise=A_noise, 
            noise_channel='flux_noise', 
            noise_operator='d_hamiltonian_d_phase', 
            evals_count=evals_count, 
            spectrum_data=spectrum_data, 
            **kwargs
        )
        
    def get_tphi_charge_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        A_noise: float = 1e-4,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the Tphi times for charge noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        A_noise : float, optional
            The amplitude of the noise (default is 1e-4).
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the Tphi calculation method.
            
        Returns
        -------
        SpectrumData
            The Tphi times for charge noise over the range of parameter values.
        """
        return self._get_tphi_vs_paramvals(
            param_name=param_name, 
            param_vals=param_vals, 
            A_noise=A_noise, 
            noise_channel='charge_noise', 
            noise_operator='d_hamiltonian_d_ng', 
            evals_count=evals_count, 
            spectrum_data=spectrum_data, 
            **kwargs
        )
        
    def get_tphi_CQPS_vs_paramvals(
        self,
        param_name: str = None,
        param_vals: np.ndarray = None,
        fp: float = 17e9,
        z: float = 0.05,
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
    ) -> SpectrumData:
        """
        Calculates the Tphi times for Coherence Quantum Phase Slip noise over a range of parameter values.
        
        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        fp : float, optional
            The plasma frequency of the junctions in the array (default is 17 GHz).
        z : float, optional
            The adimensional impedance z = Z / R_Q (default is 0.05).
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is 6).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the Tphi calculation method.
            
        Returns
        -------
        SpectrumData
            The Tphi times for Coherence Quantum Phase Slip noise over the range of parameter values.
        """
        
        if spectrum_data is None:
            spectrum_data = self.get_matelements_vs_paramvals('displacement_operator', param_name, param_vals, evals_count=evals_count)
        elif 'displacement_operator' not in spectrum_data.matrixelem_table:
            new_spec = self.get_matelements_vs_paramvals(
                'displacement_operator',
                spectrum_data.param_name,
                spectrum_data.param_vals,
                evals_count=evals_count
            )
            spectrum_data.matrixelem_table['displacement_operator'] = new_spec.matrixelem_table['displacement_operator']            
        phase_slip_frequency = 4 * np.sqrt(2)/np.pi * fp / np.sqrt(z) * np.exp(-4 / np.pi / z)
        
        displacement_operator = spectrum_data.matrixelem_table['displacement_operator']
        displacement_operator_diagonal = np.diagonal(displacement_operator, axis1=1, axis2=2)
        
        structure_factor = displacement_operator_diagonal[:,:,np.newaxis] - displacement_operator_diagonal[:,np.newaxis,:]
        
        if param_name == 'El':
            N_junctions = fp / 2 / np.pi / (param_vals * 1e9) / z
        else:
            N_junctions  = fp / 2 / np.pi / (self.El * 1e9) / z
        
        rate = np.pi * np.sqrt(N_junctions) * phase_slip_frequency * np.abs(structure_factor)
        rate[rate == 0] = np.inf
        tphi_table = 1 / rate
        
        noise_channel = 'CQPS'
        
        spectrum_data.tphi_table[noise_channel] = tphi_table
        
        return spectrum_data
        
    
    def get_tphi_vs_paramvals(
        self, 
        noise_channels: Union[str, List[str]],
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        evals_count: int = 6,
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> SpectrumData:
        
        if spectrum_data is not None:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            evals_count = spectrum_data.energy_table.shape[1]
        elif param_name is None or param_vals is None:
            raise ValueError("If spectrum_data is None, param_name and param_vals must be provided.")
        
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]
                        
        if 'flux_noise' in noise_channels:
            A_noise = kwargs.pop('A_noise', 1e-6)
            spectrum_data = self.get_tphi_flux_vs_paramvals(param_name, param_vals, A_noise, evals_count, spectrum_data, **kwargs)
            
        if 'charge_noise' in noise_channels:
            A_noise = kwargs.pop('A_noise', 1e-4)
            spectrum_data = self.get_tphi_charge_vs_paramvals(param_name, param_vals, A_noise, evals_count, spectrum_data, **kwargs)
            
        if 'CQPS' in noise_channels:
            fp = kwargs.pop('fp', 17e9)
            z = kwargs.pop('z', 0.05)
            spectrum_data = self.get_tphi_CQPS_vs_paramvals(param_name, param_vals, fp, z, evals_count, spectrum_data, **kwargs)
            
        return spectrum_data
    
    def plot_evals_vs_paramvals(
        self, 
        param_name: str = None,
        param_vals: np.ndarray = None,
        evals_count: int = 6,
        subtract_ground: bool = False,
        spectrum_data: SpectrumData = None, 
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot eigenvalues as a function of a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to vary.
        param_vals : np.ndarray
            Values of the parameter to vary.
        evals_count : int, optional
            Number of eigenvalues to calculate (default is 6).
        subtract_ground : bool, optional
            Whether to subtract the ground state energy from all eigenvalues (default is False).
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.
            - color: str or list of str, optional
                Color of the lines. Can be a single color or a list of colors.
            - linestyle: str or list of str, optional
                Linestyle of the lines. Can be a single linestyle or a list of linestyles.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if spectrum_data is None:
            if param_name is None or param_vals is None:
                raise ValueError("Both param_name and param_vals must be provided.")
            spectrum_data = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=evals_count)
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            
        evals = spectrum_data.energy_table.copy()
        if subtract_ground:
            evals -= evals[:, 0][:, np.newaxis]
            evals = evals[:, 1:] # Remove ground state

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax

        if param_name == 'phase':
            param_vals = param_vals/2/np.pi
            
        color = kwargs.get("color", None)
        linestyle = kwargs.get("linestyle", None)
        ax.plot(param_vals, evals[:,:evals_count], color=color, linestyle=linestyle)

        xlabel = self.PARAM_LABELS.get(param_name, param_name)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energy [GHz]")
        # ax.grid(True)

        return fig, ax
    
    def plot_matelem_vs_paramvals(
        self, 
        operator: str, 
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        select_elems: Union[int, List[Tuple[int, int]]] = [(1,0)], 
        mode: str = "abs",
        spectrum_data: SpectrumData = None, 
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot matrix elements as a function of a parameter.

        Parameters
        ----------
        operator : str
            Name of the operator.
        param_name : str
            Name of the parameter to vary.
        param_vals : np.ndarray
            Values of the parameter to vary.
        select_elems : Union[int, List[Tuple[int, int]]], optional
            Number of elements to select or list of specific elements to plot (default is 4).
        mode : str, optional
            Mode for plotting the matrix elements ('abs', 'real', 'imag') (default is 'abs').
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if spectrum_data is None:
            if param_name is None or param_vals is None:
                raise ValueError("Both param_name and param_vals must be provided.")
            select_elems = select_elems if isinstance(select_elems, list) else [select_elems]
            evals_count = max(max(i,j) for i, j in select_elems) + 1
            spectrum_data = self.get_matelements_vs_paramvals(operator, param_name, param_vals, evals_count=evals_count)
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax

        matrixelem_table = spectrum_data.matrixelem_table[operator]

        if isinstance(select_elems, int):
            select_elems = [(i, j) for i in range(select_elems) for j in range(i, select_elems)]

        operator_label = self.OPERATOR_LABELS.get(operator, operator)
        
        if param_name == 'phase':
            param_vals = param_vals/2/np.pi
            
        for (i, j) in select_elems:
            if mode == "abs":
                values = np.abs(matrixelem_table[:, i, j])
                ylabel = rf"$|\langle {{i}} | {operator_label} | {{j}} \rangle|$"
            elif mode == "real":
                values = matrixelem_table[:, i, j].real
                ylabel = rf"$\Re(\langle {{i}} | {operator_label} | {{j}} \rangle)$"
            elif mode == "imag":
                values = matrixelem_table[:, i, j].imag
                ylabel = rf"$\Im(\langle {{i}} | {operator_label} | {{j}} \rangle)$"
            else:
                raise ValueError(f"Unsupported mode: {mode}")

            ax.plot(param_vals, values, label=f"({i},{j})")

        xlabel = self.PARAM_LABELS.get(param_name, param_name)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        # ax.grid(True)

        return fig, ax
    
    def plot_t1_vs_paramvals(
        self,
        noise_channels: List[str],
        param_name: str = None, 
        param_vals: np.ndarray = None, 
        select_elems: Union[int, List[Tuple[int, int]]] = [(1, 0)],
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot T1 times due to capacitive noise as a function of a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to vary.
        param_vals : np.ndarray
            Values of the parameter to vary.
        select_elems : Union[int, List[Tuple[int, int]]], optional
            Number of elements to select or list of specific elements to plot (default is 4).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]
            
        if isinstance(select_elems, int):
            evals_count = select_elems
        elif isinstance(select_elems, list):
            evals_count = max(max(i,j) for i, j in select_elems) + 1
                        
        if spectrum_data is None or spectrum_data.t1_table is None or not all(channel in spectrum_data.t1_table for channel in noise_channels):
            spectrum_data = self.get_t1_vs_paramvals(noise_channels, param_name, param_vals, evals_count=evals_count, spectrum_data=spectrum_data)

        param_name = spectrum_data.param_name
        param_vals = spectrum_data.param_vals
            
        if isinstance(select_elems, int):
            select_elems = [(i, j) for i in range(select_elems) for j in range(i) if i > j]

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax
            
        rate_effective = np.zeros_like(param_vals)
        
        if param_name == 'phase':
            param_vals = param_vals/2/np.pi

        for (i, j) in select_elems:
            for channel in noise_channels:
                t1_times = spectrum_data.t1_table[channel][:, i, j]
                rate_effective += 1 / t1_times
                ax.plot(param_vals, t1_times, label=f"{channel} ({i},{j})")

        ax.plot(param_vals, 1/rate_effective, label=f"T1 effective", color='black', linestyle='--')

        xlabel = self.PARAM_LABELS.get(param_name, param_name)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$T_1$ [s]")
        ax.legend()
        # ax.grid(True)

        return fig, ax
    
    def plot_tphi_vs_paramvals(
        self, 
        noise_channels: List[str], 
        param_name: str = None, 
        param_vals: np.ndarray = None,
        select_elems: Union[int, List[Tuple[int, int]]] = [(1,0)], 
        spectrum_data: SpectrumData = None,
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Tphi times due to specified noise channels as a function of a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to vary.
        param_vals : np.ndarray
            Values of the parameter to vary.
        select_elems : Union[int, List[Tuple[int, int]]] optional
            Number of elements to select or list of specific elements to plot (default is 4).
        noise_channels : List[str], optional
            List of noise channels to consider (default is ['capacitive']).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments for plotting. Can include:
            - fig_ax: Tuple[plt.Figure, plt.Axes], optional
                Figure and axes to use for plotting. If not provided, a new figure and axes are created.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if spectrum_data is None:
            evals_count = max(max(i,j) for i, j in select_elems) + 1
            operators = set()
            for channel in noise_channels:
                if channel == 'flux_noise':
                    operators.add('d_hamiltonian_d_phase')
                if channel == 'charge_noise':
                    operators.add('d_hamiltonian_d_ng')
                else:
                    raise ValueError(f"Unsupported Tphi noise channel: {channel}")
            spectrum_data = self.get_matelements_vs_paramvals(list(operators), param_name, param_vals, evals_count=evals_count)
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals

        if isinstance(select_elems, int):
            select_elems = [(i, j) for i in range(select_elems) for j in range(i) if i > j]

        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]

        for channel in noise_channels:
            if channel not in spectrum_data.tphi_table:
                spectrum_data = self.get_tphi_vs_paramvals([channel], param_name, param_vals, spectrum_data=spectrum_data)
                
        if param_name == 'phase':
            param_vals = param_vals/2/np.pi

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax
            
        for (i, j) in select_elems:
            for channel in noise_channels:
                tphi_times = spectrum_data.tphi_table[channel][:,i,j]
                ax.plot(param_vals, tphi_times, label=f"Tphi {channel} ({i}->{j})")            

        xlabel = self.PARAM_LABELS.get(param_name, param_name)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$T_\varphi$ [s]")
        ax.legend()

        return fig, ax  
    
    def set_param(self, param_name: str, val: Any) -> None:
        """
        Sets the value of a parameter if it exists.

        Parameters
        ----------
        param_name : str
            The name of the parameter to set.
        val : Any
            The value to set for the parameter.

        Raises
        ------
        AttributeError
            If the parameter does not exist.
        """
        if not hasattr(self, param_name):
            raise AttributeError(f"Parameter '{param_name}' does not exist in the Ferbo class.")
        setattr(self, param_name, val)
        
    def _generate_suptitle(self, exclude_params: Union[str, List[str]] = None) -> str:
        """
        Generate the suptitle for the plot, excluding the specified parameters.

        Parameters
        ----------
        exclude_params : Union[str, List[str]]
            The parameter(s) to exclude from the title.

        Returns
        -------
        str
            The generated suptitle.
        """
        if isinstance(exclude_params, str):
            exclude_params = [exclude_params]
        exclude_params = exclude_params or []

        def format_value(value):
            if isinstance(value, float):
                return f'{value:.4f}'.rstrip('0').rstrip('.')
            return str(value)

        title_parts = [
            rf'{label} = {format_value(getattr(self, param))}' if param not in exclude_params else ''
            for param, label in self.PARAM_LABELS.items()
        ]
        return ', '.join(filter(None, title_parts))