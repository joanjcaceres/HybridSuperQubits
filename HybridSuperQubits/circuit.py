import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, null_space
from scipy.constants import hbar, e, h
from typing import Tuple

PHI0 = h / (2 * e)
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

    def eigenvals(self) -> np.ndarray:
        """
        Compute the eigenvalues of the dynamical matrix.
        The eigenvalues are the square of the angular frequencies (omega^2).
        They are returned in SI (rad/s)^2.
        """
        op = self.dynamical_matrix()
        evals = eigh(op, eigvals_only=True)
        return evals
    
    def eigensys(self) -> tuple:
        """
        Compute the eigenvalues and eigenvectors of the dynamical matrix.
        Returns a tuple of (eigenvalues, eigenvectors).
        The eigenvalues are the square of the angular frequencies (omega^2).
        """
        op = self.dynamical_matrix()
        evals, evecs = eigh(op)
        return evals, evecs
    
    def phase_modes(self, esys: Tuple[np.ndarray, np.ndarray] = None) -> np.ndarray:
        """
        Compute the phase modes of the circuit.
        The phase modes are obtained from the eigenvectors of the dynamical matrix.
        They are returned in units radians.
        """
        if esys is None:
            evals, evecs = self.eigensys()
        omega = np.sqrt(evals)
        M_inv = np.linalg.pinv(self.M_matrix)
        return 2 * np.pi / PHI0 * (M_inv @ self.C_eff_inv_sqrt @ evecs) * np.sqrt(hbar / 2 / omega)
    
    def resonance_frequencies(self) -> np.ndarray:
        """
        Compute the resonance frequencies of the circuit.
        The resonance frequencies are the square roots of the eigenvalues of the dynamical matrix.
        They are returned in GHz.
        """
        evals = self.eigenvals()
        return np.sqrt(evals) / (2 * np.pi * 1e9) 