import numpy as np
from scipy.constants import hbar, h, e
from typing import Optional, List, Dict, Tuple, Any
from scipy.optimize import curve_fit
from .utilities import calculate_error_metrics

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
        float: Plasma frequency in GHz.
        """
        f_p = 1 / np.sqrt(self.Lj * self.Cj) / 2 / np.pi / 1e9  # Convert to GHz
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
        
        return self.analytical_mode_frequencies(f_p, c_ratio, self.N, k)
    
    def group_velocity(self) -> float:
        """
        Calculate the group velocity of the Josephson Junction Array.

        Returns:
        np.ndarray: Group velocity in GHz for each mode.
        """
        k = np.arange(1, self.N)
        f_p = self.plasma_frequency
        c_ratio = self.C_ratio
        angle = np.pi * k / self.N
        
        v_g = np.zeros_like(k, dtype=float)
        
        # Group velocity formula
        v_g = f_p / 2 * np.sin(angle) / ((1 - np.cos(angle) + c_ratio / 2)**(3/2) * np.sqrt(1 - np.cos(angle)))
        
        return v_g

    def kerr_matrix(self) -> np.ndarray:
        """
        Compute the Kerr matrix K_{kk'} for the Josephson Junction Array using the analytical formula.
        Units in GHz.

        This implementation is based on Eq. (14) from:
        Krupko et al., "Kerr nonlinearity in a superconducting Josephson metamaterial".

        Returns
        -------
        np.ndarray
            A (N, N) matrix of Kerr coefficients.
        """
 
        # Derived quantities
        Ej = (hbar / 2 / e)**2 / self.Lj / h / 1e9 # Josephson energy (GHz)
        N = self.N
        f_k = self.resonance_frequencies()  # Angular frequencies (rad/s)
 
        # Compute f_k * f_k'
        f_outer = np.outer(f_k, f_k)
 
        # Delta_{kk'} = identity matrix
        delta = np.eye(N)
 
        # Factor: (1/2 + delta/8)
        prefactor = 0.5 + delta / 8
 
        # Final Kerr matrix
        K = prefactor * f_outer / (2 * N * Ej)
 
        return K

    @staticmethod
    def analytical_mode_frequencies(
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
            'f_p': 20,      # 20 GHz plasma frequency
            'C_ratio': 0.001  # Cg/Cj ratio (typical value)
        }
        
        param_bounds = {
            'f_p': (5, 50),       # 5-50 GHz
            'C_ratio': (0.0001, 0.1)  # Reasonable Cg/Cj range
        }
        
        return initial_params, param_bounds


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
        error_metrics = calculate_error_metrics(fitted_freqs, measured_frequencies)
        
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
    ) -> Dict[str, Any]:
        """
        Fit JJA parameters to measured resonance frequencies.
        
        Parameters:
        -----------
        measured_frequencies : np.ndarray
            Experimentally measured resonance frequencies in GHz.
        N : int
            Number of junctions in the array.
        mode_indices : List[int], optional
            Indices of the modes corresponding to the measured frequencies.
            If None, assumes consecutive modes starting from 1.
        initial_params : Dict[str, float], optional
            Initial guess for parameters. Can include 'f_p' (plasma frequency in GHz) 
            and 'C_ratio' (Cg/Cj ratio). If None, reasonable defaults will be used.
        param_bounds : Dict[str, Tuple[float, float]], optional
            Bounds for parameters as (min, max) tuples. 
            Can include 'f_p' and 'C_ratio'.
        fixed_params : Dict[str, float], optional
            Fixed parameters during fitting. Can include 'Cj' to fix the junction 
            capacitance, which allows converting fitted parameters back to Lj and Cg.
        
        Returns:
        --------
        Dict[str, Any]
            Dict with detailed results.
        """
        # Set up parameters
        if mode_indices is None:
            mode_indices = list(range(1, len(measured_frequencies) + 1))
        
        if len(mode_indices) != len(measured_frequencies):
            raise ValueError("mode_indices and measured_frequencies must have the same length")
            
        mode_indices_array = np.array(mode_indices)
        
        # Get default parameters if needed
        if initial_params is None or param_bounds is None or fixed_params is None:
            default_initial, default_bounds = JosephsonJunctionArray._get_default_params()
            
            if initial_params is None:
                initial_params = default_initial
            
            if param_bounds is None:
                param_bounds = default_bounds
            
        
        # Prepare parameters for curve_fit
        p0 = [initial_params['f_p'], initial_params['C_ratio']]
        bounds = (
            [param_bounds['f_p'][0], param_bounds['C_ratio'][0]],
            [param_bounds['f_p'][1], param_bounds['C_ratio'][1]]
        )
        
        # Define the fitting model
        def fitting_model(k, f_p, c_ratio):
            return JosephsonJunctionArray.analytical_mode_frequencies(f_p, c_ratio, N, k)
            
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
            fitted_f_p, fitted_C_ratio = popt
                        
            # Calculate parameter uncertainties
            perr = np.sqrt(np.diag(pcov))
            f_p_uncertainty, c_ratio_uncertainty = perr
            
            fitted_freqs = fitting_model(mode_indices_array, fitted_f_p, fitted_C_ratio)
            
            # Create results dictionary
            results = JosephsonJunctionArray._create_results_dict(
                fitted_f_p, fitted_C_ratio,
                f_p_uncertainty, c_ratio_uncertainty, pcov,
                fitted_freqs, measured_frequencies,
                mode_indices_array
            )
            
            return results
            
        except RuntimeError as e:
            # Handle fitting failure
            return None, {
                'fit_success': False,
                'error_message': str(e),
                'input_parameters': {
                    'initial_params': initial_params,
                    'param_bounds': param_bounds,
                    'N': N,
                    'measured_frequencies': measured_frequencies,
                    'mode_indices': mode_indices
                }
            }
