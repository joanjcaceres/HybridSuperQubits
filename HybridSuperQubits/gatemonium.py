import numpy as np
from .qubit_base import QubitBase
from scipy.linalg import sinm, cosm
from .operators import destroy, creation
from typing import Union, Tuple, Dict, Any, Iterable, Optional
import matplotlib.pyplot as plt
from scipy.integrate import quad



class Gatemonium(QubitBase):
    
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'El': r'$E_L$',
        'Ej': r'$E_J$',
        'Delta': r'$\Delta$',
        'T': r'$T$',
        'phase': r'$\Phi_{ext} / \Phi_0$'
    }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'phase_operator': r'\hat{\phi}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    'd_hamiltonian_d_phase': r'\partial \hat{H} / \partial \phi_{{ext}}',
    'd_hamiltonian_d_Ej': r'\partial \hat{H} / \partial E_J',
    }
    
    def __init__(self, Ec, El, Ej, T, phase, dimension, flux_grouping, Delta=44):
        """
        Initializes the Gatemonium class with the given parameters.

        Parameters
        ----------
        Ec : float
            Charging energy.
        El : float
            Inductive energy.
        Ej : float
            Josephson energy.
        T : Union[float, List[float], np.ndarray]
            Transmission coefficient(s). Can be a single value or a list/array for multiple channels.
        phase : float
            External phase.
        dimension : int
            Hilbert space dimension.
        flux_grouping : str
            Grouping of the external flux, either 'L' or 'ABS'.
        Delta : float, optional
            Superconducting gap. Default is 44 (for Aluminum).
        """
        
        if flux_grouping not in ['L', 'ABS']:
            raise ValueError("Invalid flux grouping; must be 'L' or 'ABS'.")
        
        self.Ec = Ec
        self.El = El
        self.Ej = Ej
        self.Delta = Delta
        
        # Convert T to numpy array for consistent handling of single or multiple channels
        self.T = np.atleast_1d(T)
        self.num_channels = len(self.T)
        
        self.phase = phase
        self.dimension = dimension
        self.flux_grouping = flux_grouping
        self.num_coef = 4
        
        # self.phi_grid = np.linspace(- 8 * np.pi, 8 * np.pi, self.dimension, endpoint=False)
        # self.dphi = self.phi_grid[1] - self.phi_grid[0] 
        
        super().__init__(dimension = dimension)
        
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
        
    def n_operator(self):
        dimension = self.dimension
        return 1j/2 * (self.El/2/self.Ec)**0.25 * (creation(dimension) - destroy(dimension))
    
    def phase_operator(self):
        dimension = self.dimension
        return (2*self.Ec/self.El)**0.25 * (creation(dimension) + destroy(dimension))
    
    def junction_potential(self):
        """
        Calculate the total junction potential operator, including both the ABS potential 
        and the standard Josephson potential.
        
        This method returns the operator form (matrix representation) of the junction potential
        for use in the Hamiltonian. For the scalar energy function, see the potential() method.
        
        The ABS potential is calculated using Fourier coefficients from the ABS energy.
        The Josephson potential is the standard -Ej*cos(phi) term.
        
        Returns
        -------
        np.ndarray
            Combined junction potential matrix in the phase basis.
        """
        phase_op = self.phase_operator()
        
        if self.flux_grouping == 'ABS':
            phase_op -= self.phase * np.eye(self.dimension)
        
        # Calculate ABS junction term using Fourier expansion for each channel
        def f(phi, T, Delta):
            return - Delta * np.sqrt(1 - T * np.sin(phi/2)**2)
        
        def A_k(k, T, Delta):
            integral, error = quad(lambda x: f(x, T, Delta) * np.cos(k*x), 0, np.pi)
            return 2 * integral / np.pi
        
        # Initialize the ABS junction term
        abs_junction_term = np.zeros((self.dimension, self.dimension), dtype=complex)
        
        # Sum contributions from all channels
        for channel_idx, T_channel in enumerate(self.T):
            # Calculate Fourier coefficients for this channel
            A_coeffs = [A_k(k, T_channel, self.Delta) for k in range(self.num_coef + 1)]
            
            # Add constant term
            abs_junction_term += A_coeffs[0] / 2 * np.eye(self.dimension)
            
            # Add cosine terms
            for k in range(1, self.num_coef + 1):
                abs_junction_term += A_coeffs[k] * cosm(k * phase_op)
        
        # Calculate standard Josephson term
        josephson_term = 0
        if self.Ej > 0:
            # Standard Josephson potential term: -Ej*cos(phi)
            josephson_term = -self.Ej * cosm(phase_op)
            
        # Combine both junction contributions
        return abs_junction_term + josephson_term
    
    def hamiltonian(self):
        """
        Calculate the Hamiltonian of the gatemonium qubit.
        
        The Hamiltonian includes:
        - Kinetic term (4*Ec*n²)
        - Inductive term (0.5*El*φ²)
        - Junction potential term (including both ABS and Josephson contributions)
        
        Returns
        -------
        np.ndarray
            The Hamiltonian matrix.
        """
        n_op = self.n_operator()
        phase_op = self.phase_operator()

        if self.flux_grouping == 'L':
            phase_op += self.phase * np.eye(self.dimension)
        
        kinetic_term = 4 * self.Ec * (n_op @ n_op)
        inductive_term = 0.5 * self.El * (phase_op @ phase_op)
        
        # Get combined junction potential (ABS + Josephson)
        junction_term = self.junction_potential()

        return kinetic_term + inductive_term + junction_term
    
    def d_hamiltonian_d_phase(self) -> np.ndarray:
        """
        Calculate the derivative of the Hamiltonian with respect to the external phase.
        
        Returns
        -------
        np.ndarray
            Derivative of the Hamiltonian with respect to the external phase.
        """
        phase_op = self.phase_operator()
        ext_phase = self.phase * np.eye(self.dimension)
        
        if self.flux_grouping == 'L':
            return self.El * (phase_op + ext_phase)
        elif self.flux_grouping == 'ABS':
            phase_op = self.phase_operator() - self.phase * np.eye(self.dimension)
            
            def f(phi, T, Delta):
                return -Delta * np.sqrt(1 - T * np.sin(phi/2)**2)
            
            def A_k(k, T, Delta):
                integral, error = quad(lambda x: f(x, T, Delta) * np.cos(k*x), 0, np.pi)
                return 2 * integral / np.pi
            
            # Initialize the derivative
            dH_dPhi = np.zeros((self.dimension, self.dimension), dtype=complex)
            
            # Calculate derivative for each channel
            for channel_idx, T_channel in enumerate(self.T):
                # Calculate Fourier coefficients for this channel
                A_coeffs = [A_k(k, T_channel, self.Delta) for k in range(0, self.num_coef + 1)]
                
                # Sum sine terms for this channel
                for k in range(1, self.num_coef + 1):
                    dH_dPhi += A_coeffs[k] * k * sinm(k * phase_op)
            
            # Add derivative of Josephson term
            if self.Ej > 0:
                dH_dPhi += self.Ej * sinm(phase_op)
                
            return dH_dPhi
            
            
        
    def d_hamiltonian_d_Ej(self) -> np.ndarray:
        """
        Calculate the derivative of the Hamiltonian with respect to Ej.

        Returns
        -------
        np.ndarray
            The derivative matrix.
        """
        phase_op = self.phase_operator()
        if self.flux_grouping == 'ABS':
            phase_op -= self.phase * np.eye(self.dimension)
        
        # Derivative of -Ej*cos(phi) with respect to Ej is -cos(phi)
        return -cosm(phase_op)
    
    def potential(self, phi: Union[float, np.ndarray]):
        """
        Calculate the potential energy as a function of phase.
        
        Parameters
        ----------
        phi : Union[float, np.ndarray]
            Phase value(s) at which to evaluate the potential.
            
        Returns
        -------
        Union[float, np.ndarray]
            Potential energy value(s).
        """
        phi_array = np.atleast_1d(phi)
        
        # Calculate inductive term
        if self.flux_grouping == 'L':
            inductive_term = 0.5 * self.El * (phi_array + self.phase)**2
        elif self.flux_grouping == 'ABS':
            inductive_term = 0.5 * self.El * phi_array**2
            
        # Calculate Josephson term
        if self.flux_grouping == 'L':
            # Josephson term with external flux in inductance
            josephson_term = -self.Ej * np.cos(phi_array)
        elif self.flux_grouping == 'ABS':
            # Josephson term with external flux in junction
            josephson_term = -self.Ej * np.cos(phi_array - self.phase)
            
        # Calculate ABS junction term for all channels
        junction_term = np.zeros_like(phi_array, dtype=float)
        for channel_idx, T_channel in enumerate(self.T):
            if self.flux_grouping == 'L':
                junction_term += -self.Delta * np.sqrt(1 - T_channel * np.sin(phi_array/2)**2)
            elif self.flux_grouping == 'ABS':
                junction_term += -self.Delta * np.sqrt(1 - T_channel * np.sin((phi_array - self.phase)/2)**2)
            
        return inductive_term + junction_term + josephson_term
        
    def wavefunction(
        self,
        which: int = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None
        ) -> Dict[str, Any]:
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
            evals, evecs = self.eigensys(evals_count)
        else:
            evals, evecs = esys
            
        dim = self.dimension
                        
        if phi_grid is None:
            phi_grid = np.linspace(-5 * np.pi, 5 * np.pi, 151)

        phi_basis_labels = phi_grid
        wavefunc_osc_basis_amplitudes = evecs[:, which]
        phi_wavefunc_amplitudes = np.zeros_like(phi_grid, dtype=np.complex128)
        
        for n in range(dim):
            phi_wavefunc_amplitudes += wavefunc_osc_basis_amplitudes[n] * self.harm_osc_wavefunction(n, phi_basis_labels, self.phase_zpf)

        return {
            "basis_labels": phi_basis_labels,
            "amplitudes": phi_wavefunc_amplitudes,
            "energy": evals[which]
        }

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
            fig.suptitle(self._generate_suptitle())
        else:
            fig, ax = fig_ax
        
        ax.plot(phi_grid/2/np.pi, potential, color='black', label='Potential')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys)
            phi_basis_labels = wavefunc_data["basis_labels"]
            wavefunc_amplitudes = wavefunc_data["amplitudes"]
            wavefunc_energy = wavefunc_data["energy"]

            ax.plot(
                phi_basis_labels/2/np.pi,
                wavefunc_energy + scaling * (wavefunc_amplitudes.real + wavefunc_amplitudes.imag),
                # color="blue",
                label=rf"$\Psi_{idx}$"
                )

        ax.set_xlabel(r"$\Phi / \Phi_0$")
        ax.set_ylabel(r"$\psi(\varphi)$, Energy [GHz]")
        ax.legend()
        ax.grid(True)

        return fig, ax


