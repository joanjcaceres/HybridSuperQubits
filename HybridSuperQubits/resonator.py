import numpy as np
import matplotlib.pyplot as plt
from .qubit_base import QubitBase
from .operators import destroy, creation
from typing import Dict, Any, Tuple, Union, Iterable, Optional
from scipy.constants import hbar, h

class Resonator(QubitBase):
    """
    Class representing a quantum LC resonator (cavity).
    
    This class models a simple harmonic oscillator, which is the fundamental
    building block for modeling quantum cavities, transmission line resonators,
    and other linear resonant structures in circuit QED.
    """
    
    PARAM_LABELS = {
        'f0': r'$f_0$',
        'frequency': r'$\omega_0/2\pi$',
        'dimension': r'$N_{\mathrm{fock}}$'
    }
    
    OPERATOR_LABELS = {
        'n_operator': r'\hat{n}',
        'a_operator': r'\hat{a}',
        'adag_operator': r'\hat{a}^\dagger',
        'position_operator': r'\hat{x}',
        'momentum_operator': r'\hat{p}',
    }
    
    def __init__(self, frequency: float, dimension: int):
        """
        Initialize a Resonator instance.
        
        Parameters
        ----------
        frequency : float
            Resonance frequency in GHz.
        dimension : int
            Dimension of the Fock space truncation.
        """
        self.frequency = frequency  # GHz
        self.dimension = dimension
        super().__init__(self.dimension)
    
    @property
    def f0(self) -> float:
        """
        Returns the resonance frequency.
        
        Returns
        -------
        float
            Resonance frequency in GHz.
        """
        return self.frequency
    
    @property
    def omega0(self) -> float:
        """
        Returns the angular frequency.
        
        Returns
        -------
        float
            Angular frequency in rad/s.
        """
        return 2 * np.pi * self.frequency * 1e9
    
    @property
    def zpf_voltage(self) -> float:
        """
        Returns the zero-point fluctuation voltage scale.
        For a transmission line resonator with characteristic impedance Z0.
        
        Returns
        -------
        float
            Zero-point fluctuation voltage scale.
        """
        Z0 = 50  # Ohm, typical transmission line impedance
        return np.sqrt(hbar * self.omega0 * Z0 / 2)
    
    @property
    def zpf_current(self) -> float:
        """
        Returns the zero-point fluctuation current scale.
        
        Returns
        -------
        float
            Zero-point fluctuation current scale.
        """
        Z0 = 50  # Ohm
        return np.sqrt(hbar * self.omega0 / (2 * Z0))
    
    def n_operator(self) -> np.ndarray:
        """
        Returns the number operator.
        
        Returns
        -------
        np.ndarray
            The number operator matrix.
        """
        return np.diag(np.arange(self.dimension, dtype=float))
    
    def a_operator(self) -> np.ndarray:
        """
        Returns the annihilation operator.
        
        Returns
        -------
        np.ndarray
            The annihilation operator matrix.
        """
        return destroy(self.dimension)
    
    def adag_operator(self) -> np.ndarray:
        """
        Returns the creation operator.
        
        Returns
        -------
        np.ndarray
            The creation operator matrix.
        """
        return creation(self.dimension)
    
    def position_operator(self) -> np.ndarray:
        """
        Returns the position operator (dimensionless).
        
        For a transmission line resonator, this corresponds to the 
        voltage fluctuations normalized by the zero-point voltage.
        
        Returns
        -------
        np.ndarray
            The position operator matrix.
        """
        a = self.a_operator()
        adag = self.adag_operator()
        return (a + adag) / np.sqrt(2)
    
    def momentum_operator(self) -> np.ndarray:
        """
        Returns the momentum operator (dimensionless).
        
        For a transmission line resonator, this corresponds to the 
        current fluctuations normalized by the zero-point current.
        
        Returns
        -------
        np.ndarray
            The momentum operator matrix.
        """
        a = self.a_operator()
        adag = self.adag_operator()
        return 1j * (adag - a) / np.sqrt(2)
    
    def hamiltonian(self) -> np.ndarray:
        """
        Returns the Hamiltonian for the resonator.
        
        H = ħω₀(a†a + 1/2)
        
        Returns
        -------
        np.ndarray
            The Hamiltonian matrix in GHz units.
        """
        n_op = self.n_operator()
        return self.frequency * (n_op + 0.5 * np.eye(self.dimension))
    
    def thermal_state(self, temperature: float) -> np.ndarray:
        """
        Calculate the thermal state density matrix.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        
        Returns
        -------
        np.ndarray
            Thermal state density matrix.
        """
        from scipy.constants import k
        
        kT = k * temperature / h  # Convert to frequency units (Hz)
        kT_GHz = kT / 1e9  # Convert to GHz
        
        # Calculate thermal occupation number
        if kT_GHz > 0:
            n_thermal = 1 / (np.exp(self.frequency / kT_GHz) - 1)
        else:
            n_thermal = 0
        
        # Calculate thermal state probabilities
        probs = np.zeros(self.dimension)
        for n in range(self.dimension):
            if n_thermal > 0:
                probs[n] = (n_thermal / (1 + n_thermal))**n * (1 / (1 + n_thermal))
            else:
                probs[n] = 1.0 if n == 0 else 0.0
        
        # Normalize probabilities
        probs = probs / np.sum(probs)
        
        # Create thermal density matrix
        rho_thermal = np.diag(probs)
        return rho_thermal
    
    def coherent_state(self, alpha: complex) -> np.ndarray:
        """
        Generate a coherent state |α⟩.
        
        Parameters
        ----------
        alpha : complex
            Coherent state amplitude.
        
        Returns
        -------
        np.ndarray
            Coherent state vector.
        """
        # Coherent state coefficients: ⟨n|α⟩ = e^(-|α|²/2) α^n / √(n!)
        from scipy.special import factorial
        
        coeffs = np.zeros(self.dimension, dtype=complex)
        normalization = np.exp(-np.abs(alpha)**2 / 2)
        
        for n in range(self.dimension):
            coeffs[n] = normalization * (alpha**n) / np.sqrt(factorial(n))
        
        return coeffs
    
    def fock_state(self, n: int) -> np.ndarray:
        """
        Generate a Fock state |n⟩.
        
        Parameters
        ----------
        n : int
            Fock state number.
        
        Returns
        -------
        np.ndarray
            Fock state vector.
        """
        if n >= self.dimension:
            raise ValueError(f"Fock state number {n} exceeds dimension {self.dimension}")
        
        state = np.zeros(self.dimension)
        state[n] = 1.0
        return state
    
    def wavefunction(
        self, 
        which: int = 0,
        x_grid: np.ndarray = None,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        ) -> Dict[str, Any]:
        """
        Calculate the wavefunction in position representation.
        
        Parameters
        ----------
        which : int, optional
            Index of the eigenstate (default is 0 for ground state).
        x_grid : np.ndarray, optional
            Position grid for wavefunction evaluation.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Eigenvalues and eigenvectors. If None, they will be calculated.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing position grid, wavefunction amplitudes, and energy.
        """
        if esys is None:
            evals, evecs = self.eigensys()
        else:
            evals, evecs = esys
        
        if x_grid is None:
            # Default position grid for harmonic oscillator
            x_max = np.sqrt(2 * (which + 1) + 4)  # Adaptive grid based on state
            x_grid = np.linspace(-x_max, x_max, 151)
        
        # Get the Fock state amplitudes
        fock_amplitudes = evecs[:, which]
        
        # Calculate position wavefunction using harmonic oscillator wavefunctions
        psi_x = np.zeros(len(x_grid), dtype=complex)
        
        for n in range(self.dimension):
            if np.abs(fock_amplitudes[n]) > 1e-10:  # Skip negligible amplitudes
                psi_n = self.harm_osc_wavefunction(n, x_grid, 1.0)
                psi_x += fock_amplitudes[n] * psi_n
        
        return {
            "basis_labels": x_grid,
            "amplitudes": psi_x,
            "energy": evals[which]
        }
    
    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0,
        x_grid: np.ndarray = None,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        scaling: Optional[float] = 1,
        mode: str = 'abs',
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the wavefunction(s) in position representation.
        
        Parameters
        ----------
        which : Union[int, Iterable[int]], optional
            Index or indices of the eigenstates to plot.
        x_grid : np.ndarray, optional
            Position grid for wavefunction evaluation.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Eigenvalues and eigenvectors.
        scaling : Optional[float], optional
            Scaling factor for wavefunction amplitude display.
        mode : str, optional
            Display mode: 'abs', 'real', or 'imag'.
        **kwargs
            Additional plotting arguments.
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        if isinstance(which, int):
            which = [which]
        
        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle())
        else:
            fig, ax = fig_ax
        
        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, x_grid=x_grid, esys=esys)
            x_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]
            
            if mode == 'abs':
                y_values = np.abs(wavefunc_amplitudes)
            elif mode == 'real':
                y_values = wavefunc_amplitudes.real
            elif mode == 'imag':
                y_values = wavefunc_amplitudes.imag
            else:
                raise ValueError("Invalid mode; must be 'abs', 'real', or 'imag'.")
            
            ax.plot(
                x_basis_labels,
                wavefunc_energy + scaling * y_values,
                label=rf"$|{idx}\rangle$"
            )
        
        ax.set_xlabel(r"Position $x$ (dimensionless)")
        ax.set_ylabel(r"$\psi(x)$, Energy [GHz]")
        ax.legend()
        ax.grid(True)
        
        return fig, ax
    
    def plot_fock_state_populations(
        self, 
        state: np.ndarray, 
        **kwargs
        ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the Fock state populations of a given state.
        
        Parameters
        ----------
        state : np.ndarray
            State vector or density matrix.
        **kwargs
            Additional plotting arguments.
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle("Fock State Populations")
        else:
            fig, ax = fig_ax
        
        # Extract populations
        if state.ndim == 1:  # State vector
            populations = np.abs(state)**2
        else:  # Density matrix
            populations = np.real(np.diag(state))
        
        n_states = np.arange(len(populations))
        ax.bar(n_states, populations, alpha=0.7)
        ax.set_xlabel("Fock State Number $n$")
        ax.set_ylabel("Population $|c_n|^2$")
        ax.set_xticks(n_states)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def potential(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate the harmonic oscillator potential.
        
        Parameters
        ----------
        x : Union[float, np.ndarray]
            Position coordinate(s).
        
        Returns
        -------
        Union[float, np.ndarray]
            Potential energy in GHz units.
        """
        return 0.5 * self.frequency * x**2