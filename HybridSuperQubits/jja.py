import numpy as np
from scipy.sparse import diags
from scipy.linalg import eigh
from typing import Optional, List, Dict, Tuple, Any
from scipy.optimize import curve_fit

class JosephsonJunctionArray:
    
    def __init__(self, Lj: float, Cj: float, Cg: float, N: int):
        """
        Initialize the Josephson Junction Array with the given parameters.

        Parameters:
        Lj (float): Inductance of the junctions in Henries.
        Cj (float): Capacitance of the junctions in Farads.
        Cg (float): Capacitance to ground in Farads.
        N (int): Number of junctions in the array.
        """
        self.Lj = Lj
        self.Cj = Cj
        self.Cg = Cg
        self.N = N
        
    @property
    def plasma_frequency(self) -> float:
        """
        Calculate the plasma frequency of the Josephson Junction Array.

        Returns:
        float: Plasma frequency in Hz.
        """
        f_p = 1 / np.sqrt(self.Lj * self.Cj) / 2 / np.pi
        return f_p
    
    @property
    def C_ratio(self) -> float:
        """
        Calculate the capacitance ratio Cg/Cj.

        Returns:
        float: Capacitance ratio.
        """
        return self.Cg / self.Cj
    
    def resonance_frequencies(self) -> np.ndarray:
        """
        Calculate the resonance frequencies in Hz of the Josephson Junction Array.

        Returns:
        --------
        np.ndarray
            Resonance frequencies in Hz.
        """
        k = np.arange(1, self.N + 1)
        f_p = self.plasma_frequency
        c_ratio = self.C_ratio
        
        return self.calculate_resonance_frequencies(f_p, c_ratio, self.N, k)

    @staticmethod
    def calculate_resonance_frequencies(
        f_p: float, 
        c_ratio: float, 
        N: int, 
        k: Optional[np.ndarray] = None
        ) -> np.ndarray:
        """
        Static helper method to calculate resonance frequencies from parameters.
        
        Parameters:
        -----------
        f_p : float
            Plasma frequency in Hz.
        c_ratio : float
            Capacitance ratio (Cg/Cj).
        N : int
            Number of junctions in the array.
        k : np.ndarray, optional
            Indices of the modes to calculate. If None, all modes from 1 to N are calculated.
            
        Returns:
        --------
        np.ndarray
            Resonance frequencies in Hz.
        """
        if k is None:
            k = np.arange(1, N + 1)
            
        return f_p * np.sqrt(1 / (1 + c_ratio / 2 / (1 - np.cos(np.pi * k / N))))

    
    @staticmethod
    def _get_default_params() -> Tuple[Dict, Dict, Dict]:
        """
        Get default parameter values for fitting.
        
        Parameters:
        -----------
        measured_frequencies : np.ndarray
            Used to determine default mode indices if needed
        
        Returns:
        --------
        Tuple[Dict, Dict, Dict]
            Default initial parameters, bounds, and fixed parameters
        """
        initial_params = {
            'f_p': 20e9,      # 20 GHz plasma frequency
            'C_ratio': 0.002  # Cg/Cj ratio (typical value)
        }
        
        param_bounds = {
            'f_p': (5e9, 50e9),       # 5-50 GHz
            'C_ratio': (0.0001, 0.1)  # Reasonable Cg/Cj range
        }
        
        fixed_params = {
            'Cj': 50e-15  # 50 fF (typical junction capacitance)
        }
        
        return initial_params, param_bounds, fixed_params

    @staticmethod
    def _calculate_error_metrics(fitted_freqs: np.ndarray, measured_frequencies: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate error metrics between fitted and measured frequencies.
        
        Parameters:
        -----------
        fitted_freqs : np.ndarray
            Fitted frequency values
        measured_frequencies : np.ndarray
            Measured frequency values
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with different error metrics
        """
        residuals = fitted_freqs - measured_frequencies
        rel_error = residuals / measured_frequencies * 100
        rms_error = np.sqrt(np.mean(residuals**2))
        std_dev = np.std(residuals)
        max_error = np.max(np.abs(residuals))
        
        return {
            'residuals': residuals,
            'relative_error_percent': rel_error,
            'rms_error_Hz': rms_error,
            'std_dev_Hz': std_dev,
            'max_error_Hz': max_error
        }

    @staticmethod
    def _create_results_dict(
        fitted_f_p: float, 
        fitted_C_ratio: float, 
        f_p_uncertainty: float,
        c_ratio_uncertainty: float,
        pcov: np.ndarray,
        fitted_freqs: np.ndarray,
        measured_frequencies: np.ndarray,
        mode_indices_array: np.ndarray
    ) -> Dict[str, Any]:
        """
        Create a comprehensive results dictionary from fitted parameters.
        
        Parameters:
        -----------
        Various fitted parameters and uncertainties
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive results dictionary
        """
        error_metrics = JosephsonJunctionArray._calculate_error_metrics(fitted_freqs, measured_frequencies)
        
        return {
            'parameters': {
                'f_p': fitted_f_p,
                'c_ratio': fitted_C_ratio,
            },
            'uncertainties': {
                'f_p': f_p_uncertainty,
                'c_ratio': c_ratio_uncertainty,
                'covariance_matrix': pcov
            },
            'frequencies': {
                'fitted': fitted_freqs,
                'measured': measured_frequencies,
                'mode_indices': mode_indices_array
            },
            'errors': error_metrics,
            'fit_success': True
        }

    @staticmethod
    def fit_from_measurements(
        measured_frequencies: np.ndarray,
        N: int,
        mode_indices: Optional[List[int]] = None, 
        initial_params: Optional[Dict[str, float]] = None, 
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fixed_params: Optional[Dict[str, float]] = None
    ) -> Tuple['JosephsonJunctionArray', Dict[str, Any]]:
        """
        Fit JJA parameters to measured resonance frequencies.
        
        Parameters:
        -----------
        measured_frequencies : np.ndarray
            Experimentally measured resonance frequencies in Hz.
        N : int
            Number of junctions in the array.
        mode_indices : List[int], optional
            Indices of the modes corresponding to the measured frequencies.
            If None, assumes consecutive modes starting from 1.
        initial_params : Dict[str, float], optional
            Initial guess for parameters. Can include 'f_p' (plasma frequency in Hz) 
            and 'C_ratio' (Cg/Cj ratio). If None, reasonable defaults will be used.
        param_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for parameters as (min, max) tuples. 
            Can include 'f_p' and 'C_ratio'.
        fixed_params : Dict[str, float], optional
            Fixed parameters during fitting. Can include 'Cj' to fix the junction 
            capacitance, which allows converting fitted parameters back to Lj and Cg.
        
        Returns:
        --------
        Tuple[JosephsonJunctionArray, Dict[str, Any]]
            (JJA instance with fitted parameters, dict with detailed results)
        """
        # Set up parameters
        if mode_indices is None:
            mode_indices = list(range(1, len(measured_frequencies) + 1))
        
        if len(mode_indices) != len(measured_frequencies):
            raise ValueError("mode_indices and measured_frequencies must have the same length")
            
        mode_indices_array = np.array(mode_indices)
        
        # Get default parameters if needed
        if initial_params is None or param_bounds is None or fixed_params is None:
            default_initial, default_bounds, default_fixed = JosephsonJunctionArray._get_default_params()
            
            if initial_params is None:
                initial_params = default_initial
            
            if param_bounds is None:
                param_bounds = default_bounds
            
            if fixed_params is None:
                fixed_params = default_fixed
        
        # Prepare parameters for curve_fit
        p0 = [initial_params['f_p'], initial_params['C_ratio']]
        bounds = (
            [param_bounds['f_p'][0], param_bounds['C_ratio'][0]],
            [param_bounds['f_p'][1], param_bounds['C_ratio'][1]]
        )
        
        # Define the fitting model
        def fitting_model(k, f_p, c_ratio):
            return JosephsonJunctionArray.calculate_resonance_frequencies(f_p, c_ratio, N, k)
            
        # Perform the curve fit
        try:
            # Execute curve fit
            popt, pcov = curve_fit(
                fitting_model,
                mode_indices_array, 
                measured_frequencies,
                p0=p0,
                bounds=bounds,
                method='trf'
            )
            
            # Extract fitted parameters
            fitted_f_p = popt[0]
            fitted_C_ratio = popt[1]
            
            # Calculate parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            f_p_uncertainty = perr[0]
            c_ratio_uncertainty = perr[1]
            
            # Convert to physical parameters
            Cj = fixed_params['Cj']
            Lj = 1/((2*np.pi*fitted_f_p)**2 * Cj)
            Cg = fitted_C_ratio * Cj
            
            # Create JJA instance with fitted parameters
            fitted_jja = JosephsonJunctionArray(Lj, Cj, Cg, N)
            
            fitted_freqs = fitting_model(mode_indices_array, fitted_f_p, fitted_C_ratio)
            
            # Create results dictionary
            results = JosephsonJunctionArray._create_results_dict(
                fitted_f_p, fitted_C_ratio,
                f_p_uncertainty, c_ratio_uncertainty, pcov,
                fitted_freqs, measured_frequencies,
                mode_indices_array
            )
            
            return fitted_jja, results
            
        except RuntimeError as e:
            # Handle fitting failure
            return None, {
                'fit_success': False,
                'error_message': str(e),
                'input_parameters': {
                    'initial_params': initial_params,
                    'param_bounds': param_bounds,
                    'fixed_params': fixed_params,
                    'N': N,
                    'measured_frequencies': measured_frequencies,
                    'mode_indices': mode_indices
                }
            }
        


def L_inv_matrix(N,Ljj):
    matrix_diagonals = [(-1/Ljj)*np.ones(N),(2/Ljj)*np.ones(N+1),(-1/Ljj)*np.ones(N)]
    matrix = diags(matrix_diagonals,[-1,0,1]).toarray()
    matrix[0,0] = 1/Ljj
    matrix[-1,-1] = 1/Ljj
    return matrix

def jja_resonances(params):
    Ljj = params[0]*1e-9
    Cjj = params[1]*1e-15
    Cg = params[2]*1e-18
    Cg_big = params[3]*1e-15
    Cin = params[4]*1e-15
    
    N = 170
    Cout = 0

    # RESOLVER EL PROBLEMA DE REDONDEO DE NUMEROS PEQUEÑOS.

    # Eigensolution of the linearized circuit matrix for a Josephson junction chain with the following parameters:
    # N: number of junctions
    # Cjj: junction capacitance
    # Ljj: junction inductance
    # Cg: ground capacitance
    # Cin: input capacitance
    # Cout: output capacitance

    # Return the frequency in Hz.

    # References:
    #   https://theses.hal.science/tel-01369020
    #   https://journals.aps.org/prb/abstract/10.1103/PhysRevB.92.104508

    # Solving C^-1/2 L^-1 C^-1/2 phi = \omega^2 phi

    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cg_big,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals = eigh(matrix_operation, eigvals_only=True) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    return np.sqrt(eigvals)/2/np.pi

def jja_eigensys(params, **kwargs):
    Ljj = params[0]*1e-9
    Cjj = params[1]*1e-15
    Cg = params[2]*1e-18
    Cg_big = params[3]*1e-15
    Cin = params[4]*1e-15
    
    N = 170
    Cout = 0

    eigenvalues_C, eigenvectors_C = eigh(C_matrix(N,Cjj,Cg,Cg_big,Cin,Cout))
    Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigenvalues_C))
    C_inv_sqrt = np.dot(eigenvectors_C, np.dot(Lambda_inv_sqrt, eigenvectors_C.T)) # spectral decomposition of C^-1/2
    matrix_operation = np.dot(C_inv_sqrt, np.dot(L_inv_matrix(N,Ljj), C_inv_sqrt)) # C^-1/2 L^-1 C^-1/2
    eigvals, eigvecs = eigh(matrix_operation, **kwargs) # eigenvalues and eigvecs of C^-1/2 L^-1 C^-1/2

    return np.sqrt(eigvals), eigvecs # Returns \omega_k, \Phi_k





