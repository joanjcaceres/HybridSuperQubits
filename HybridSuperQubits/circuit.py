import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, null_space, expm
from scipy.constants import hbar, e, h
import scipy.sparse as sp
import functools

PHI0 = hbar / (2 * e)
class Circuit:
    def __init__(
        self,
        C_matrix: np.ndarray,
        cycle_matrix: np.ndarray,
        L_inv_matrix: np.ndarray,
    ):
        self.C_matrix = C_matrix
        self.L_inv_matrix = L_inv_matrix
        
        # Validate that C_matrix and L_inv_matrix are symmetric positive definite
        if not np.allclose(C_matrix, C_matrix.T):
            raise ValueError("C_matrix must be symmetric.")
        if not np.allclose(L_inv_matrix, L_inv_matrix.T):
            raise ValueError("L_inv_matrix must be symmetric.")
        
        self.cycle_matrix = np.atleast_2d(cycle_matrix)
        self.N = self.cycle_matrix.shape[1] - self.cycle_matrix.shape[0] # Number of dynamical variables
        self.C_inv = self._C_inv()
        self.M_matrix = self._M_matrix()
        self.M_augmented_inv = self._M_augmented_inv()
        self.C_eff_inv_sqrt = self._C_eff_inv_sqrt()
        self.L_inv_eff, self.L_inv_cycle = self._L_inv_eff_matrix()
        # Cache eigen-decomposition of dynamical matrix for reuse
        self._eigenvals2, self._eigenvecs = eigh(self.dynamical_matrix())
        self._omega = np.sqrt(self._eigenvals2)  # mode frequencies in rad/s
        
    def _C_inv(self) -> np.ndarray:
        """
        Compute the inverse of the capacitance matrix (C^(-1)).
        """
        c, lower = cho_factor(self.C_matrix)
        C_inv = cho_solve((c, lower), np.eye(self.C_matrix.shape[0]))
        return C_inv
    
    def _M_matrix(self) -> np.ndarray:
        """
        Compute the reduction matrix (M_+).
        """
        M_matrix = null_space(self.cycle_matrix @ self.C_inv).T
        return M_matrix

    def _M_augmented_inv(self) -> np.ndarray:
        """
        Compute the reduction matrix (M_+^{-1}).
        """
        M_augmented = np.vstack((self.M_matrix, self.cycle_matrix))
        M_augmented_inv = np.linalg.inv(M_augmented)

        return M_augmented_inv
    
    def _C_eff_inv_sqrt(self) -> np.ndarray:
        """
        Compute the inverse square root of the capacitance matrix (C^(-1/2)).
        This is used in the dynamical matrix calculation.
        """
        C_eff_matrix = self.M_augmented_inv.T @ self.C_matrix @ self.M_augmented_inv
        eigvals_C, eigvecs_C = eigh(C_eff_matrix)
        # clamp eigenvalues to avoid numerical issues
        eigvals_C = np.clip(eigvals_C, 1e-15, None)
        Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigvals_C))
        total_matrix = eigvecs_C @ Lambda_inv_sqrt @ eigvecs_C.T
        return total_matrix[:self.N, :self.N]
    
    def _L_inv_eff_matrix(self) -> np.ndarray:
        """
        Compute the effective inverse inductance matrix (L_inv_eff).
        The effective inverse inductance matrix is defined as:
        L_inv_eff = M_+^{-1} * L_inv * M_+^{-1}
        where M is the reduction matrix.
        The first N rows and columns correspond to the dynamical variables
        The last rows and columns correspond to the cycle variables
        """
        total_matrix = self.M_augmented_inv.T @ self.L_inv_matrix @ self.M_augmented_inv
        return total_matrix[:self.N, :self.N], total_matrix[self.N:, :self.N]
    
    def dynamical_matrix(self) -> np.ndarray:
        """
        Compute the dynamical matrix for the circuit.
        The dynamical matrix is defined as:
        D = C^(-1/2) * L^(-1) * C^(-1/2)
        where C is the capacitance matrix and L is the inductance matrix.
        """
        return self.C_eff_inv_sqrt @ self.L_inv_eff @ self.C_eff_inv_sqrt

    def _matrix_cos(self, op: np.ndarray) -> np.ndarray:
        """
        Compute the matrix cosine via exponentials.
        """
        return (expm(1j * op) + expm(-1j * op)).real * 0.5

    @functools.lru_cache(maxsize=None)
    def creation_operator(self, dim: int) -> sp.csr_matrix:
        """
        Sparse creation operator of given Hilbert-space dimension.
        """
        data = np.sqrt(np.arange(1, dim))
        return sp.diags([data], [1], shape=(dim, dim), format='csr')

    @functools.lru_cache(maxsize=None)
    def annihilation_operator(self, dim: int) -> sp.csr_matrix:
        """
        Sparse annihilation operator of given Hilbert-space dimension.
        """
        data = np.sqrt(np.arange(1, dim))
        return sp.diags([data], [-1], shape=(dim, dim), format='csr')

    @functools.lru_cache(maxsize=None)
    def phase_operator(self, dim: int, mode_idx: int) -> sp.csr_matrix:
        """
        Cached phase operator φ̂ = sqrt(ħ/(2ω)) (a + a†) for given mode.
        """
        a = self.annihilation_operator(dim)
        adag = self.creation_operator(dim)
        factor = np.sqrt(hbar / (2 * self._omega[mode_idx]))
        return factor * (a + adag)

    def hamiltonian_0(self, dim: int, mode_idx: int = 0) -> sp.csr_matrix:
        """
        Diagonal harmonic Hamiltonian H0 = ħω (n + 1/2), returned in GHz units.
        """
        # freq in GHz
        freq_ghz = self._omega[mode_idx] / (2 * np.pi * 1e9)
        n = np.arange(dim)
        diag = (n + 0.5) * freq_ghz
        return sp.diags([diag], [0], format='csr')

    def hamiltonian_1(self, dim: int, phi_ext: float, mode_idx: int = 0) -> sp.csr_matrix:
        """
        Linear coupling Hamiltonian H1 = φ_ext * (Φ0/h) * coupling * φ̂ (in GHz).
        """
        phase_op = self.phase_operator(dim, mode_idx)
        # coupling coefficient from circuit structure
        coupling = (self.L_inv_cycle @ self.C_eff_inv_sqrt @ self._eigenvecs)[mode_idx, mode_idx]
        # scale by flux quantum, external flux, and convert to GHz
        scale = PHI0 / h / 1e9 * phi_ext
        return scale * coupling * phase_op

    def hamiltonian_nl(
        self,
        dim: int,
        Ej: float,  # Josephson energy in GHz
        phi_ext: float,
        mode_idx: int = 0
    ) -> sp.csr_matrix:
        """
        Nonlinear Josephson Hamiltonian H_nl = -Ej [cos(φ̂+φ_ext)+½(φ̂+φ_ext)²] (in GHz).
        """
        # build phase operator
        phase_op = self.phase_operator(dim, mode_idx)
        # displacement due to mode and circuit
        weight = (self.M_augmented_inv[mode_idx, :-1] @ self.C_eff_inv_sqrt @ self._eigenvecs)[mode_idx]
        phi_x_0 = weight * phase_op / PHI0
        phi_total = phi_x_0 + phi_ext * self.M_augmented_inv[mode_idx, -1] * sp.eye(dim, format='csr')
        # Convert to dense for matrix functions
        phi_dense = phi_total.toarray()
        eigvals_phi, eigvecs_phi = eigh(phi_dense)
        # calculte f(phi) = cos(phi) + 1/2 * phi^2
        f_vals = np.cos(eigvals_phi) + 0.5 * eigvals_phi**2
        H_nl = -Ej * (eigvecs_phi @ np.diag(f_vals) @ eigvecs_phi.T)
        return sp.csr_matrix(H_nl)