import numpy as np
from scipy.linalg import eigh

class Circuit:
    def __init__(
        self,
        C_matrix: np.ndarray,
        L_inv_matrix: np.ndarray,
    ):
        self.C_matrix = C_matrix
        self.L_inv_matrix = L_inv_matrix
        

    def C_inv_sqrt(self) -> np.ndarray:
        """
        Compute the inverse square root of the capacitance matrix (C^(-1/2)).
        This is used in the dynamical matrix calculation.
        """
        eigvals_C, eigvecs_C = eigh(self.C_matrix)
        Lambda_inv_sqrt = np.diag(1 / np.sqrt(eigvals_C))
        return eigvecs_C @ Lambda_inv_sqrt @ eigvecs_C.T
        
    def dynamical_matrix(self) -> np.ndarray:
        """
        Compute the dynamical matrix for the circuit.
        The dynamical matrix is defined as:
        D = C^(-1/2) * L^(-1) * C^(-1/2)
        where C is the capacitance matrix and L is the inductance matrix.
        """
        return self.C_inv_sqrt() @ self.L_inv_matrix @ self.C_inv_sqrt()

    def eigenvals(self) -> np.ndarray:
        op = self.dynamical_matrix()
        evals = eigh(op, eigvals_only=True)
        return evals
    
    def flux_modes(self) -> np.ndarray:
        op = self.dynamical_matrix()
        _, evecs = eigh(op)
        return (self.C_inv_sqrt() @ evecs).T[1:]
    
    def resonance_frequencies(self) -> np.ndarray:
        """
        Compute the resonance frequencies of the circuit.
        The resonance frequencies are the square roots of the eigenvalues of the dynamical matrix.
        They are returned in GHz.
        """
        evals = self.eigenvals()
        return np.sqrt(evals) / (2 * np.pi * 1e9)

    
    # @staticmethod
    # def fit_from_measurements(
    #     measured_frequencies: np.ndarray,
    #     N: int,
    #     build_C_matrix_fn: Callable[['Circuit'], np.ndarray],
    #     build_L_inv_matrix_fn: Callable[['Circuit'], np.ndarray],
    #     initial_params: Optional[List[float]] = None,
    #     bounds: Optional[Tuple[List[float], List[float]]] = None,
    #     relative_error: bool = False,
    #     verbose: bool = True,
    #     extra_param_names: Optional[List[str]] = None,
    #     build_extra_params_fn: Optional[Callable[[List[float]], Dict[str, float]]] = None
    # ) -> Tuple['Circuit', Dict[str, Any]]:
    #     """
    #     Fit JJA parameters to measured resonance frequencies using numerical matrix-based model.
    #     Parameters
    #     ----------
    #     measured_frequencies : np.ndarray
    #         Experimentally measured resonance frequencies in Hz.
    #     N : int
    #         Number of junctions in the array.
    #     build_C_matrix_fn : Callable
    #         Function to build the capacitance matrix.
    #     build_L_inv_matrix_fn : Callable
    #         Function to build the inverse inductance matrix.
    #     initial_params : List[float], optional
    #         Initial guess for [Lj (nH), Cj (fF), Cg (aF)].
    #     bounds : Tuple[List[float], List[float]], optional
    #         Bounds for the parameters as ([min_vals], [max_vals]).
    #     Returns
    #     -------
    #     Tuple[Circuit, Dict[str, Any]]
    #         Fitted instance and results dictionary.
    #     """
    #     if initial_params is None:
    #         initial_params = [1.0, 30.0, 50.0]  # Lj [nH], Cj [fF], Cg [aF]
        
    #     def create_jja_instance(params: List[float]) -> Circuit:
    #         base_params = params[:3]
    #         extra_param_values = params[3:] if extra_param_names else []
    #         extra_params = build_extra_params_fn(extra_param_values) if build_extra_params_fn else {}

    #         Lj, Cj, Cg = base_params[0]*1e-9, base_params[1]*1e-15, base_params[2]*1e-18

    #         return Circuit(
    #             Lj, Cj, Cg, N,
    #             build_C_matrix_fn, build_L_inv_matrix_fn,
    #             extra_params=extra_params
    #         )
        
    #     def model(params):
    #         return create_jja_instance(params).resonance_frequencies()[:len(measured_frequencies)]
        
    #     def cost(params):
    #         diff = model(params) - measured_frequencies
    #         if relative_error:
    #             return diff / measured_frequencies
    #         return diff
        
    #     from scipy.optimize import least_squares
    #     result = least_squares(cost, initial_params, bounds=bounds, loss='soft_l1')
    #     jja_fitted = create_jja_instance(result.x)
    #     residuals = result.fun
    #     measured = measured_frequencies

    #     if verbose:
    #         print("Fit summary:")
    #         print("Parameters:", result.x)
    #         print("Cost:", result.cost)
    #         print("Message:", result.message)

    #     param_dict = {
    #         "Lj_nH": result.x[0],
    #         "Cj_fF": result.x[1],
    #         "Cg_aF": result.x[2],
    #     }

    #     if extra_param_names:
    #         for i, name in enumerate(extra_param_names):
    #             param_dict[name] = result.x[3 + i]

    #     return jja_fitted, {
    #         "parameters": param_dict,
    #         "frequencies": {
    #             "measured": measured,
    #             "fitted": jja_fitted.resonance_frequencies()[:len(measured)]
    #         },
    #         "errors": {
    #             "rmse": float(np.sqrt(np.mean(residuals**2))),
    #             "r_squared": float(1 - np.sum(residuals**2) / np.sum((measured - np.mean(measured))**2))
    #         },
    #         "residuals": residuals,
    #         "fit_success": result.success,
    #         "cost": result.cost,
    #         "message": result.message
    #     }