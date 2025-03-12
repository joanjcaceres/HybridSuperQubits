import matplotlib.pyplot as plt
import numpy as np
from .qubit_base import QubitBase
from typing import Any, Dict, Optional, Tuple, Union, Iterable
from .operators import sigma_z, sigma_y, sigma_x
from .utilities import sin_kphi_operator, cos_kphi_operator

class Andreev(QubitBase):
    PARAM_LABELS = {
        'Ec': r'$E_C$',
        'Gamma': r'$\Gamma$',
        'delta_Gamma': r'$\delta \Gamma$',
        'er': r'$\epsilon_r$',
        'phase': r'$\Phi_{ext} / \Phi_0$',
        'ng': r'$n_g$'
    }
    
    OPERATOR_LABELS = {
    'n_operator': r'\hat{n}',
    'd_hamiltonian_d_ng': r'\partial \hat{H} / \partial n_g',
    'd_hamiltonian_d_deltaGamma': r'\partial \hat{H} / \partial \delta \Gamma',
    'd_hamiltonian_d_er': r'\partial \hat{H} / \partial \epsilon_r',
    }
    
    def __init__(self, Ec, Gamma, delta_Gamma, er, phase, ng, n_cut, Delta = 40):
        """
        Initializes the Ferbo class with the given parameters.

        Parameters
        ----------
        Ec : float
            Charging energy.
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
        ng : float
            Charge offset.
        n_cut : int
            Maximum number of charge states.
        Delta : float
            Superconducting gap.
        """
        
        self.Ec = Ec
        self.Gamma = Gamma
        self.delta_Gamma = delta_Gamma
        self.er = er
        self.phase = phase
        self.ng = ng
        self.n_cut = n_cut
        self.dimension = 2 * (self.n_cut * 4 + 1)
        self.Delta = Delta
        
        super().__init__(self.dimension)
        
    
    def n_operator(self) -> np.ndarray:
        """
        Returns the charge number operator adjusted for half-charge translations.

        Returns
        -------
        np.ndarray
            The charge number operator.
        """
        n_values = np.arange(-self.n_cut, self.n_cut+1/2 , 1/2)
        n_matrix = np.diag(n_values)
        return np.kron(np.eye(2), n_matrix)  
    
    def jrl_potential(self) -> np.ndarray:
        """
        Returns the Josephson Resonance Level potential in the half-charge basis.

        Returns
        -------
        np.ndarray
            The Josephson Resonance Level potential.
        """
        
        Gamma_term = -self.Gamma * np.kron(sigma_z(), cos_kphi_operator(1, self.dimension // 2, self.phase/2))
        delta_Gamma_term = - self.delta_Gamma * np.kron(sigma_y(), sin_kphi_operator(1, self.dimension // 2, self.phase/2))
        e_r_term = self.er * np.kron(sigma_x(), np.eye(self.dimension // 2))
        
        return Gamma_term + delta_Gamma_term + e_r_term
            
    # def zazunov_potential(self) -> np.ndarray:
        
    def hamiltonian(self) -> np.ndarray:
        """
        Returns the Hamiltonian of the system.

        Returns
        -------
        np.ndarray
            The Hamiltonian of the system.
        """
        n_x = self.delta_Gamma/4/(self.Gamma+self.Delta)
        n_op = self.n_operator() + n_x * np.kron(sigma_x(),np.eye(self.dimension//2)) - self.ng * np.kron(np.eye(2),np.eye(self.dimension//2))
        
        charge_term = 4 * self.Ec * n_op @ n_op

        potential = self.jrl_potential()
        return charge_term + potential
    
    def d_hamiltonian_d_ng(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the number of charge offset.
        
        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the number of charge offset.
        
        """
        n_x = self.delta_Gamma/4/(self.Gamma+self.Delta)
        n_op = self.n_operator() + n_x * np.kron(sigma_x(),np.eye(self.dimension//2)) - self.ng * np.kron(np.eye(2),np.eye(self.dimension//2))

        return  - 8 * self.Ec * n_op
    
    def d_hamiltonian_d_phase(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the external magnetic phase.

        Returns
        -------
        np.ndarray
            The derivative of the Hamiltonian with respect to the external magnetic phase.
        """
        return NotImplementedError("Not implemented yet")
                
    def d_hamiltonian_d_er(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the energy relaxation rate.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the energy relaxation rate.
        """
        # return - np.kron(np.eye(self.dimension // 2),sigma_z())
        return NotImplementedError("Not implemented yet")
    
    def d_hamiltonian_d_deltaGamma(self) -> np.ndarray:
        """
        Returns the derivative of the Hamiltonian with respect to the coupling strength difference.

        Returns
        -------
        Qobj
            The derivative of the Hamiltonian with respect to the coupling strength difference.
        """
        return NotImplementedError("Not implemented yet")
        # if self.flux_grouping == 'L':
        #     phase_op = self.phase_operator()[::2,::2]
        # else:
        #     phase_op = self.phase_operator()[::2,::2] - self.phase * np.eye(self.dimension // 2)
        # return - np.kron(sinm(phase_op/2),sigma_y())

    def numberbasis_wavefunction(
        self, 
        which: int = 0,
        esys: Tuple[np.ndarray, np.ndarray] = None
        ) -> Dict[str, Any]:
        """
        Returns a wave function in the number basis.

        Parameters
        ----------
        which : int, optional
            Index of desired wave function (default is 0).
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors (default is None).

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
            
        dim = self.dimension // 2
        evecs = evecs.T
                        
        n_grid = np.arange(-self.n_cut, self.n_cut + 1/2, 1/2)
        wf_up = evecs[which, :dim]
        wf_down = evecs[which, dim:]
        number_wavefunc_amplitudes = np.vstack((wf_up, wf_down))

        return {
            "basis_labels": n_grid,
            "amplitudes": number_wavefunc_amplitudes,
            "energy": evals[which]
        }
        
    def wavefunction(
        self, 
        which: int = 0,
        phi_grid: np.ndarray = None,
        esys: Tuple[np.ndarray, np.ndarray] = None,
        basis: str = 'default'
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
            Basis in which to return the wave function ('default' or 'abs') (default is 'default').

        Returns
        -------
        Dict[str, Any]
            Wave function data containing basis labels, amplitudes, and energy.
        """
        return NotImplementedError("Not implemented yet")
        
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
        return NotImplementedError("Not implemented yet")


    def plot_wavefunction(
        self, 
        which: Union[int, Iterable[int]] = 0, 
        phi_grid: np.ndarray = None, 
        esys: Tuple[np.ndarray, np.ndarray] = None, 
        scaling: Optional[float] = None,
        basis: str = 'default',
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
        basis: str, optional
            Basis in which to return the wavefunction ('default' or 'abs') (default is 'default').
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
        
        ax.plot(phi_grid/2/np.pi, potential[:, 0], color='black', label='Potential')
        ax.plot(phi_grid/2/np.pi, potential[:, 1], color='black')

        for idx in which:
            wavefunc_data = self.wavefunction(which=idx, phi_grid=phi_grid, esys=esys, basis = basis)
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