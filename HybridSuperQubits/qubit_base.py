# import scqubits.utils.plotting as plot
from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh, expm
from scipy.special import factorial, pbdv
from tqdm.notebook import tqdm

from . import noise as _noise
from .storage import SpectrumData

R_Q = _noise.R_Q  # Superconducting resistance quantum


class QubitBase(ABC):
    PARAM_LABELS: dict[str, str] = {}
    OPERATOR_LABELS: dict[str, str] = {}

    def __init__(self, dimension: int):
        self.dimension = dimension

    def __repr__(self) -> str:
        """
        Returns a string representation of the QubitBase instance.

        Returns
        -------
        str
            A string representation of the QubitBase instance.
        """
        init_params = [param for param in self.__dict__ if not param.startswith("_")]
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
        return expm(-1j * 2 * np.pi * self.n_operator()).astype(complex)

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

    def eigensys(
        self, evals_count: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates eigenvalues and corresponding eigenvectors using scipy.linalg.eigh.

        Parameters
        ----------
        evals_count : int, optional
            Number of desired eigenvalues/eigenstates (default is None, in which case all are calculated).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Eigenvalues and eigenvectors as numpy arrays.
        """
        if evals_count is None:
            evals_count = self.dimension

        hamiltonian_mat = self.hamiltonian()
        evals, evecs = eigh(
            hamiltonian_mat,
            eigvals_only=False,
            subset_by_index=(0, evals_count - 1),
            # check_finite=False,
        )
        return evals.astype(float), evecs.astype(complex)

    def eigenvals(self, evals_count: Optional[int] = None) -> np.ndarray:
        """
        Calculates eigenvalues using scipy.linalg.eigh.

        Parameters
        ----------
        evals_count : int, optional
            Number of desired eigenvalues (default is None, in which case all are calculated).

        Returns
        -------
        np.ndarray
            Eigenvalues as a numpy array.
        """
        if evals_count is None:
            evals_count = self.dimension

        hamiltonian_mat = self.hamiltonian()
        evals = eigh(
            hamiltonian_mat,
            eigvals_only=True,
            subset_by_index=(0, evals_count - 1),
            check_finite=False,
        )
        return np.sort(evals).astype(float)

    def get_spectrum_vs_paramvals(
        self,
        param_name: str,
        param_vals: Union[list[float], np.ndarray],
        evals_count: Optional[int] = None,
        subtract_ground: bool = False,
        show_progress: bool = True,
    ) -> SpectrumData:
        """
        Calculates the eigenenergies and eigenstates for a range of parameter values.

        Parameters
        ----------
        param_name : str
            The name of the parameter to vary.
        param_vals : List[float]
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is None, in which case all are calculated).
        subtract_ground : bool, optional
            Whether to subtract the ground state energy from the eigenenergies (default is False).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The eigenenergies and eigenstates for the range of parameter values.
        """
        if evals_count is None:
            evals_count = self.dimension

        eigenenergies_array = []
        eigenstates_array = []

        initial_value = getattr(self, param_name)

        for val in tqdm(param_vals, leave=False, disable=not show_progress):
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
            state_table=np.array(eigenstates_array),
        )

        if subtract_ground:
            spectrum_data.subtract_ground()

        return spectrum_data

    def matrixelement_table(
        self,
        operator: str,
        evecs: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
    ) -> np.ndarray:
        """
        Returns a table of matrix elements for a given operator with respect to the eigenstates.

        Parameters
        ----------
        operator : str
            The name of the operator.
        evecs : np.ndarray, optional
            The eigenstates (default is None, in which case they are calculated).
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is None, in which case all are calculated).

        Returns
        -------
        np.ndarray
            The table of matrix elements.
        """
        if evals_count is None:
            evals_count = self.dimension

        if evecs is None:
            _, evecs = self.eigensys(evals_count)

        operator_matrix = getattr(self, operator)()
        matrix_elements = evecs.conj().T @ operator_matrix @ evecs
        return matrix_elements.astype(complex)

    def get_matelements_vs_paramvals(
        self,
        operators: Union[str, list[str]],
        param_name: str,
        param_vals: np.ndarray,
        evals_count: Optional[int] = None,
        show_progress: bool = True,
    ) -> SpectrumData:
        # TODO: #9 Add spectrum_data as optional parameter in case it was already computed the esys.
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
            The number of eigenvalues and eigenstates to calculate (default is None, in which case all are calculated).
        show_progress : bool, optional
            Whether to display a progress bar during calculation (default is True).

        Returns
        -------
        Dict[str, Dict[str, np.ndarray]]
            The matrix elements for the operators over the range of parameter values.
        """
        if evals_count is None:
            evals_count = self.dimension

        if isinstance(operators, str):
            operators = [operators]

        paramvals_count = len(param_vals)
        eigenenergies_array = np.empty((paramvals_count, evals_count))
        eigenstates_array = np.empty(
            (paramvals_count, self.dimension, evals_count), dtype=np.complex128
        )
        matrixelem_tables = {
            operator: np.empty(
                (paramvals_count, evals_count, evals_count), dtype=complex
            )
            for operator in operators
        }

        initial_value = getattr(self, param_name)

        for idx, val in enumerate(
            tqdm(param_vals, leave=False, disable=not show_progress)
        ):
            self.set_param(param_name, val)
            eigenenergies, eigenstates = self.eigensys(evals_count)
            eigenenergies_array[idx] = eigenenergies
            eigenstates_array[idx] = eigenstates

            for operator in operators:
                matrix_elements = self.matrixelement_table(
                    operator, evecs=eigenstates, evals_count=evals_count
                )
                matrixelem_tables[operator][idx] = matrix_elements

        self.set_param(param_name, initial_value)

        spectrum_data = SpectrumData(
            energy_table=eigenenergies_array,
            system_params=self.__dict__,
            param_name=param_name,
            param_vals=param_vals,
            state_table=eigenstates_array,
            matrixelem_table=matrixelem_tables,
        )

        return spectrum_data

    def harm_osc_wavefunction(
        self, n: int, x: Union[float, np.ndarray], l_osc: float
    ) -> Union[float, np.ndarray]:
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
        return np.asarray(result[0], dtype=float)

    # def plot_matrixelements(
    #     self,
    #     operator: str,
    #     evecs: Optional[np.ndarray] = None,
    #     evals_count: int = 6,
    #     mode: str = "abs",
    #     show_numbers: bool = False,
    #     show_colorbar: bool = True,
    #     show3d: bool = True,
    #     **kwargs,
    # ) -> Union[Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]], Tuple[plt.Figure, plt.Axes]]:
    #     """
    #     Plots the matrix elements for a given operator with respect to the eigenstates.

    #     Parameters
    #     ----------
    #     operator : str
    #         The name of the operator.
    #     evecs : np.ndarray, optional
    #         The eigenstates (default is None, in which case they are calculated).
    #     evals_count : int, optional
    #         The number of eigenvalues and eigenstates to calculate (default is 6).
    #     mode : str, optional
    #         The mode for displaying matrix elements ('abs', 'real', 'imag') (default is 'abs').
    #     show_numbers : bool, optional
    #         Whether to show the matrix element values as numbers (default is False).
    #     show_colorbar : bool, optional
    #         Whether to show a colorbar (default is True).
    #     show3d : bool, optional
    #         Whether to show a 3D plot (default is True).

    #     Returns
    #     -------
    #     Union[Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]], Tuple[plt.Figure, plt.Axes]]
    #         The figure and axes of the plot.
    #     """
    #     # Obtener la tabla de elementos de la matriz utilizando el método matrixelement_table
    #     matrix_elements = self.matrixelement_table(operator, evecs, evals_count)

    #     modefunction = {
    #         "abs": np.abs,
    #         "real": np.real,
    #         "imag": np.imag
    #     }.get(mode, None)

    #     if modefunction is None:
    #         raise ValueError(f"Unsupported mode: {mode}")

    #     matrix_elements = modefunction(matrix_elements)

    #     if show3d:
    #         return plot.matrix(
    #             matrix_elements,
    #             mode=mode,
    #             show_numbers=show_numbers,
    #             **kwargs,
    #         )

    #     return plot.matrix2d(
    #         matrix_elements,
    #         mode=mode,
    #         show_numbers=show_numbers,
    #         show_colorbar=show_colorbar,
    #         **kwargs,
    #     )

    def t1_capacitive(
        self,
        i: int = 1,
        j: int = 0,
        Q_cap: Optional[Union[float, Callable]] = None,
        T: float = 0.015,
        esys: Optional[tuple[np.ndarray, np.ndarray]] = None,
        matrix_elements: Optional[np.ndarray] = None,
        get_rate: bool = False,
    ) -> float:
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table(
                "n_operator", evecs=evecs, evals_count=max(i, j) + 1
            )
        return _noise.t1_capacitive(
            evals=evals,
            n_op_matelems=matrix_elements,
            Ec=getattr(self, "Ec", 1.0),
            T=T,
            Q_cap=Q_cap,
            i=i,
            j=j,
            get_rate=get_rate,
        )

    def t1_inductive(
        self,
        i: int = 1,
        j: int = 0,
        Q_ind: Optional[float] = None,
        T: float = 0.015,
        esys: Optional[tuple[np.ndarray, np.ndarray]] = None,
        matrix_elements: Optional[np.ndarray] = None,
        get_rate: bool = False,
    ) -> float:
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table(
                "phase_operator", evecs=evecs, evals_count=max(i, j) + 1
            )
        return _noise.t1_inductive(
            evals=evals,
            phase_op_matelems=matrix_elements,
            El=getattr(self, "El", 1.0),
            T=T,
            Q_ind=Q_ind,
            i=i,
            j=j,
            get_rate=get_rate,
        )

    def t1_flux_bias_line(
        self,
        i: int = 1,
        j: int = 0,
        M: float = 2500,
        Z: float = 50,
        T: float = 0.015,
        esys: Optional[tuple[np.ndarray, np.ndarray]] = None,
        matrix_elements: Optional[np.ndarray] = None,
        get_rate: bool = False,
    ) -> float:
        if esys is None:
            evals, evecs = self.eigensys(evals_count=max(i, j) + 1)
        else:
            evals, evecs = esys
        if matrix_elements is None:
            matrix_elements = self.matrixelement_table(
                "d_hamiltonian_d_phase", evecs=evecs, evals_count=max(i, j) + 1
            )
        return _noise.t1_flux_bias_line(
            evals=evals,
            dH_dphase_matelems=matrix_elements,
            M=M,
            Z=Z,
            T=T,
            i=i,
            j=j,
            get_rate=get_rate,
        )

    def tphi_1_over_f(
        self,
        A_noise: float,
        noise_op: Union[str, list[str]],
        esys: Optional[tuple[np.ndarray, np.ndarray]] = None,
        get_rate: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Calculates the 1/f dephasing time (or rate) due to an arbitrary noise source.

        Parameters
        ----------
        A_noise : float
            Noise strength.
        noise_op : Union[str, List[str]]
            Noise operator(s) to use.
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors (default is None).
        get_rate : bool, optional
            Whether to return the rate instead of the Tphi time (default is False).

        Returns
        -------
        np.ndarray
            The 1/f dephasing time (or rate) due to an arbitrary noise source.
        """
        p = {"omega_ir": 2 * np.pi * 1, "omega_uv": 3 * 2 * np.pi * 1e9, "t_exp": 10e-6}
        p.update(kwargs)

        if esys is None:
            evals, evecs = self.eigensys()
        else:
            evals, evecs = esys

        if isinstance(noise_op, str):
            noise_op = [noise_op]

        dH_d_lambda = self.matrixelement_table(noise_op[0], evecs=evecs)
        d2H_op = getattr(self, noise_op[1])() if len(noise_op) > 1 else None

        return _noise.tphi_1_over_f(
            evals=evals,
            dH_dlambda_matelems=dH_d_lambda,
            A_noise=A_noise,
            d2H_dlambda2_op=d2H_op,
            omega_ir=p["omega_ir"],
            omega_uv=p["omega_uv"],
            t_exp=p["t_exp"],
            get_rate=get_rate,
        )

    def tphi_CQPS(
        self,
        fp: float = 17e9,
        z: float = 0.05,
        esys: Optional[tuple[np.ndarray, np.ndarray]] = None,
        get_rate: bool = False,
    ) -> np.ndarray:
        """
        Calculates the CQPS dephasing time (or rate).

        Parameters
        ----------
        fp : float
            Plasma frequency.
        z : float
            Normalized impedance (z = Z / RQ).
        esys : Tuple[np.ndarray, np.ndarray], optional
            Precomputed eigenvalues and eigenvectors (default is None).
        get_rate : bool, optional
            Whether to return the rate instead of the Tphi time (default is False).

        Returns
        -------
        np.ndarray
            The CQPS dephasing time (or rate).
        """
        if esys is None:
            evals, evecs = self.eigensys()
        else:
            evals, evecs = esys

        displacement_operator_melem = self.matrixelement_table(
            "displacement_operator", evecs=evecs
        )

        return _noise.tphi_CQPS(
            evals=evals,
            displacement_op_matelems=displacement_operator_melem,
            El=getattr(self, "El", 1.0),
            fp=fp,
            z=z,
            get_rate=get_rate,
        )

    def get_t1_vs_paramvals(
        self,
        noise_channels: Union[str, list[str]],
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
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
            The number of eigenvalues and eigenstates to calculate (default is None, in which case all are calculated).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        **kwargs
            Additional arguments to pass to the T1 calculation method.

        Returns
        -------
        SpectrumData
            The T1 times for the specified noise channels over the range of parameter values.
        """
        if evals_count is None:
            evals_count = self.dimension

        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]

        if spectrum_data is not None:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            evals_count = spectrum_data.energy_table.shape[1]
        elif param_name is None or param_vals is None:
            raise ValueError(
                "If spectrum_data is None, param_name and param_vals must be provided."
            )

        if "capacitive" in noise_channels:
            spectrum_data = self.get_t1_capacitive_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "inductive" in noise_channels:
            spectrum_data = self.get_t1_inductive_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "charge_impedance" in noise_channels:
            spectrum_data = self.get_t1_charge_impedance_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "critical_current" in noise_channels:
            spectrum_data = self.get_t1_critical_current_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "flux_bias_line" in noise_channels:
            spectrum_data = self.get_t1_flux_bias_line_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "flux_noise" in noise_channels:
            spectrum_data = self.get_t1_1_over_f_flux_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )
        if "Andreev" in noise_channels:
            spectrum_data = self.get_t1_er_vs_paramvals(
                param_name, param_vals, evals_count, spectrum_data, **kwargs
            )

        if spectrum_data is None:
            raise ValueError(
                "No valid noise channels provided or spectrum_data could not be computed."
            )

        return spectrum_data

    def get_t1_capacitive_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        Q_cap: Optional[Union[float, Callable]] = None,
        T: float = 0.015,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for capacitive noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )
        Q_cap_fun = _noise._resolve_q_factor(Q_cap, _noise.default_Q_cap)
        Ec = getattr(self, "Ec", 1.0)

        def spectral_density(omega, T):
            return _noise.S_capacitive(omega, T, Ec, Q_cap_fun)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "n_operator", "capacitive", T, **kwargs,
        )

    def get_t1_inductive_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        Q_ind: Optional[float] = None,
        T: float = 0.015,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for inductive noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )
        Q_ind_fun = _noise._resolve_q_factor(
            Q_ind, _noise.default_Q_ind_factory(T),
        )
        El = getattr(self, "El", 1.0)

        def spectral_density(omega, T):
            return _noise.S_inductive(omega, T, El, Q_ind_fun)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "phase_operator", "inductive", T, **kwargs,
        )

    def get_t1_charge_impedance_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        Z: float = 50,
        T: float = 0.015,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for charge-impedance noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )

        def spectral_density(omega, T):
            return _noise.S_charge_impedance(omega, T, Z)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "n_operator", "charge_impedance", T, **kwargs,
        )

    def get_t1_flux_bias_line_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        M: float = 2500,
        Z: float = 50,
        T: float = 0.015,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for flux-bias-line noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )

        def spectral_density(omega, T):
            return _noise.S_flux_bias_line(omega, T, M, Z)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "d_hamiltonian_d_phase", "flux_bias_line",
            T, **kwargs,
        )

    def get_t1_1_over_f_flux_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        A_noise: float = 1e-6,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for 1/f flux noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )

        def spectral_density(omega, T):
            return _noise.S_one_over_f_flux(omega, T, A_noise)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "d_hamiltonian_d_phase", "flux_noise",
            T=0.015, **kwargs,
        )

    def get_t1_critical_current_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        A_noise: float = 1e-7,
        N: int = 100,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for critical-current noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )
        El = getattr(self, "El", 1.0)

        def spectral_density(omega, T):
            return _noise.S_critical_current(omega, T, A_noise, El, N)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "d_hamiltonian_d_EL", "critical_current",
            T=0.015, **kwargs,
        )

    def get_t1_er_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        R: float = 50,
        T: float = 0.015,
        **kwargs,
    ) -> SpectrumData:
        """T1 times for Fermi-level (Andreev) noise over a parameter sweep."""
        param_name, param_vals, evals_count, spectrum_data = (
            self._resolve_sweep_inputs(
                param_name, param_vals, evals_count, spectrum_data,
            )
        )

        def spectral_density(omega, T):
            return _noise.S_andreev(omega, T, R)

        return self._get_t1_vs_paramvals(
            param_name, param_vals, evals_count, spectrum_data,
            spectral_density, "d_hamiltonian_d_er", "Andreev", T, **kwargs,
        )

    def _resolve_sweep_inputs(
        self,
        param_name: Optional[str],
        param_vals: Optional[np.ndarray],
        evals_count: Optional[int],
        spectrum_data: Optional[SpectrumData],
    ) -> tuple[str, np.ndarray, int, SpectrumData]:
        """Common boilerplate for sweep methods: resolve inputs and ensure
        ``spectrum_data`` is populated with the eigensystem table."""
        if evals_count is None:
            evals_count = self.dimension
        if spectrum_data is not None:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            evals_count = spectrum_data.energy_table.shape[1]
        elif param_name is None or param_vals is None:
            raise ValueError(
                "If spectrum_data is None, param_name and param_vals must be provided."
            )
        assert param_name is not None
        assert param_vals is not None
        if spectrum_data is None:
            spectrum_data = self.get_spectrum_vs_paramvals(
                param_name, param_vals, evals_count=evals_count,
            )
        return param_name, param_vals, evals_count, spectrum_data

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
        **kwargs,
    ) -> SpectrumData:
        """Thin wrapper: ensure ``noise_operator`` matrix elements are in
        ``spectrum_data``, then delegate to
        :func:`HybridSuperQubits.noise.t1_table_from_spectral_density`."""
        del kwargs  # accepted for caller compatibility; no remaining knobs
        if noise_operator not in spectrum_data.matrixelem_table or (
            spectrum_data.matrixelem_table[noise_operator].shape[1]
            != spectrum_data.matrixelem_table[noise_operator].shape[2]
        ):
            new_spec = self.get_matelements_vs_paramvals(
                noise_operator,
                spectrum_data.param_name,
                spectrum_data.param_vals,
                evals_count=evals_count,
            )
            spectrum_data.matrixelem_table.update(new_spec.matrixelem_table)

        spectrum_data.t1_table[noise_channel] = (
            _noise.t1_table_from_spectral_density(
                evals_table=spectrum_data.energy_table,
                matelems_table=spectrum_data.matrixelem_table[noise_operator],
                spectral_density=spectral_density,
                T=T,
            )
        )
        return spectrum_data

    def _get_tphi_1_over_f_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        A_noise: float = 2 * np.pi * 1e-6,
        noise_channel: str = "",
        noise_operators: Union[str, list[str]] = "",
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> SpectrumData:
        """Tphi (1/f) sweep wrapper: gather derivatives, delegate to
        :func:`HybridSuperQubits.noise.tphi_1_over_f_table`."""
        if evals_count is None:
            evals_count = self.dimension

        p = {"omega_ir": 2 * np.pi * 1, "omega_uv": 3 * 2 * np.pi * 1e9, "t_exp": 10e-6}
        p.update(kwargs)

        if isinstance(noise_operators, str):
            noise_operators = [noise_operators]

        first_op = noise_operators[0]
        if "d_hamiltonian_d_" not in first_op:
            raise ValueError("The operator must be of the form 'd_hamiltonian_d_X'")
        deriv_param = first_op.split("d_hamiltonian_d_")[1]

        if spectrum_data is not None:
            if param_name is None:
                param_name = spectrum_data.param_name
            if param_vals is None:
                param_vals = spectrum_data.param_vals
        if param_name is None or param_vals is None:
            raise ValueError(
                "param_name and param_vals must be provided either as arguments "
                "or through spectrum_data"
            )

        if spectrum_data is None:
            spectrum_data = self.get_matelements_vs_paramvals(
                noise_operators, param_name, param_vals, evals_count=evals_count,
            )
        elif not all(op in spectrum_data.matrixelem_table for op in noise_operators):
            missing = [
                op for op in noise_operators if op not in spectrum_data.matrixelem_table
            ]
            new_spec = self.get_matelements_vs_paramvals(
                missing, param_name, param_vals, evals_count=evals_count,
            )
            spectrum_data.matrixelem_table.update(new_spec.matrixelem_table)
        elif all(
            spectrum_data.matrixelem_table[op].shape[1]
            != spectrum_data.matrixelem_table[op].shape[2]
            for op in noise_operators
        ):
            new_spec = self.get_matelements_vs_paramvals(
                noise_operators, param_name, param_vals, evals_count=evals_count,
            )
            for op in noise_operators:
                spectrum_data.matrixelem_table[op] = new_spec.matrixelem_table[op]

        dE_d_lambda = np.diagonal(
            spectrum_data.matrixelem_table[noise_operators[0]], axis1=1, axis2=2,
        )

        if len(noise_operators) > 1:
            spectrum_data = self.get_d2E_d_param_vs_paramvals(
                operators=noise_operators, spectrum_data=spectrum_data,
            )
            d2E_table = spectrum_data.d2E_table[f"d2E_d_{deriv_param}2"]
        else:
            d2E_table = None

        spectrum_data.tphi_table[noise_channel] = _noise.tphi_1_over_f_table(
            dE_d_lambda_table=dE_d_lambda,
            A_noise=A_noise,
            d2E_d_lambda2_table=d2E_table,
            omega_ir=p["omega_ir"],
            omega_uv=p["omega_uv"],
            t_exp=p["t_exp"],
        )
        return spectrum_data

    def get_tphi_flux_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        A_noise: float = 2 * np.pi * 1e-6,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> SpectrumData:
        """
        Calculates the Tphi times for flux noise over a range of parameter values.

        Parameters
        ----------
        param_name : str, optional
            The name of the parameter to vary. If not provided, extracted from spectrum_data.
        param_vals : np.ndarray, optional
            The values of the parameter to vary. If not provided, extracted from spectrum_data.
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
        return self._get_tphi_1_over_f_vs_paramvals(
            param_name=param_name,
            param_vals=param_vals,
            A_noise=A_noise,
            noise_channel="flux_noise",
            noise_operators=["d_hamiltonian_d_phase", "d2_hamiltonian_d_phase2"],
            evals_count=evals_count,
            spectrum_data=spectrum_data,
            **kwargs,
        )

    def get_tphi_charge_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        A_noise: float = 1e-4 * 1e-9 / 4 / R_Q,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
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
        return self._get_tphi_1_over_f_vs_paramvals(
            param_name=param_name,
            param_vals=param_vals,
            A_noise=A_noise,
            noise_channel="charge_noise",
            noise_operators=["d_hamiltonian_d_er"],
            evals_count=evals_count,
            spectrum_data=spectrum_data,
            **kwargs,
        )

    def get_tphi_CQPS_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        fp: float = 17e9,
        z: float = 0.05,
        evals_count: int = 6,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
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
            if param_name is None or param_vals is None:
                raise ValueError("param_name and param_vals cannot be None")
            spectrum_data = self.get_matelements_vs_paramvals(
                "displacement_operator", param_name, param_vals, evals_count=evals_count,
            )
        elif "displacement_operator" not in spectrum_data.matrixelem_table:
            if spectrum_data.param_name is None or spectrum_data.param_vals is None:
                raise ValueError(
                    "spectrum_data.param_name and spectrum_data.param_vals cannot be None"
                )
            new_spec = self.get_matelements_vs_paramvals(
                "displacement_operator",
                spectrum_data.param_name,
                spectrum_data.param_vals,
                evals_count=evals_count,
            )
            spectrum_data.matrixelem_table["displacement_operator"] = (
                new_spec.matrixelem_table["displacement_operator"]
            )

        El_values = (
            np.asarray(param_vals, dtype=float) if param_name == "El"
            else np.full(spectrum_data.energy_table.shape[0], getattr(self, "El", 1.0))
        )

        spectrum_data.tphi_table["CQPS"] = _noise.tphi_cqps_table(
            displacement_op_matelems_table=spectrum_data.matrixelem_table[
                "displacement_operator"
            ],
            El_values=El_values,
            fp=fp,
            z=z,
        )
        return spectrum_data

    def get_tphi_vs_paramvals(
        self,
        noise_channels: Union[str, list[str]],
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> SpectrumData:
        if spectrum_data is not None:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            evals_count = spectrum_data.energy_table.shape[1]
        elif param_name is None or param_vals is None:
            raise ValueError(
                "If spectrum_data is None, param_name and param_vals must be provided."
            )

        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]

        if "flux_noise" in noise_channels:
            A_noise = kwargs.pop("A_noise", 1e-6)
            # Validate parameters before calling get_tphi_flux_vs_paramvals

            if param_name is None or param_vals is None:
                raise ValueError("param_name and param_vals cannot be None")

            spectrum_data = self.get_tphi_flux_vs_paramvals(
                param_name, param_vals, A_noise, evals_count, spectrum_data, **kwargs
            )

        if "charge_noise" in noise_channels:
            A_noise = kwargs.pop("A_noise", 1e-4)
            spectrum_data = self.get_tphi_charge_vs_paramvals(
                param_name, param_vals, A_noise, evals_count, spectrum_data, **kwargs
            )

        if "CQPS" in noise_channels:
            fp = kwargs.pop("fp", 17e9)
            z = kwargs.pop("z", 0.05)
            # Validate parameters before calling get_tphi_CQPS_vs_paramvals

            if param_name is None or param_vals is None:
                raise ValueError("param_name and param_vals cannot be None")

            if evals_count is None:
                raise ValueError("evals_count cannot be None")

            spectrum_data = self.get_tphi_CQPS_vs_paramvals(
                param_name, param_vals, fp, z, evals_count, spectrum_data, **kwargs
            )

        return spectrum_data

    def get_d2E_d_param_vs_paramvals(
        self,
        operators: list[str],
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: Optional[int] = None,
        spectrum_data: Optional[SpectrumData] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> SpectrumData:
        """
        Calculates the second derivative of energy with respect to a parameter over a range of parameter values.

        Parameters
        ----------
        operators : List[str]
            The operators used to calculate the derivatives. Must contain two elements:
            First element: first derivative operator (e.g., 'd_hamiltonian_d_phase')
            Second element: second derivative operator (e.g., 'd2_hamiltonian_d_phase2')
        param_name : str, optional
            The name of the parameter to vary.
        param_vals : np.ndarray, optional
            The values of the parameter to vary.
        evals_count : int, optional
            The number of eigenvalues and eigenstates to calculate (default is None, which uses self.dimension).
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use (default is None).
        show_progress : bool, optional
            Whether to show a progress bar during the calculation (default is True).

        Returns
        -------
        SpectrumData
            The spectral data with added second derivatives.
        """
        if evals_count is None:
            evals_count = self.dimension

        if len(operators) != 2:
            raise ValueError(
                "operators must contain exactly two elements: first and second derivative operators"
            )

        first_op = operators[0]
        if "d_hamiltonian_d_" in first_op:
            deriv_param = first_op.split("d_hamiltonian_d_")[1]
        else:
            raise ValueError(
                "The operators must be of the form 'd_hamiltonian_d_X' and 'd2_hamiltonian_d_X2'."
            )

        # Check if spectrum_data has enough eigenvalues
        if (
            spectrum_data is not None
            and spectrum_data.energy_table.shape[1] < self.dimension
        ):
            print(
                f"Warning: spectrum_data contains only {spectrum_data.energy_table.shape[1]} eigenvalues, but {self.dimension} are needed for accurate second derivative calculation."
            )
            print("Recalculating spectrum_data with sufficient eigenvalues...")
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals
            spectrum_data = None

        if spectrum_data is None:
            # Validate parameters before calling get_matelements_vs_paramvals

            if param_name is None or param_vals is None:
                raise ValueError("param_name and param_vals cannot be None")

            spectrum_data = self.get_matelements_vs_paramvals(
                operators,
                param_name,
                param_vals,
                evals_count=evals_count,
                show_progress=show_progress,
            )
        elif not all(op in spectrum_data.matrixelem_table for op in operators):
            missing_operators = [
                op for op in operators if op not in spectrum_data.matrixelem_table
            ]
            new_spec = self.get_matelements_vs_paramvals(
                missing_operators,
                param_name,
                param_vals,
                evals_count=evals_count,
                show_progress=show_progress,
            )
            spectrum_data.matrixelem_table.update(new_spec.matrixelem_table)

        param_vals = spectrum_data.param_vals

        # Calculate diagonal elements of the second derivative operator
        d2H_d_lambda2 = np.diagonal(
            spectrum_data.matrixelem_table[operators[1]], axis1=1, axis2=2
        )

        # Calculate energy differences
        E_diff = (
            spectrum_data.energy_table[:, :, np.newaxis]
            - spectrum_data.energy_table[:, np.newaxis, :]
        )
        E_diff = np.where(E_diff == 0, np.inf, E_diff)

        # Calculate the second derivative correction
        dH_d_lambda_matelems_square = (
            np.abs(spectrum_data.matrixelem_table[operators[0]]) ** 2
        )
        d2E_d_lambda2_correction = 2 * np.sum(
            dH_d_lambda_matelems_square / E_diff, axis=2
        )

        # Calculate the total second derivative
        d2E_d_lambda2 = d2H_d_lambda2 + d2E_d_lambda2_correction

        spectrum_data.d2E_table[f"d2E_d_{deriv_param}2"] = np.real(d2E_d_lambda2)

        return spectrum_data

    def plot_evals_vs_paramvals(
        self,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        evals_count: int = 6,
        subtract_ground: bool = False,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
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
            spectrum_data = self.get_spectrum_vs_paramvals(
                param_name, param_vals, evals_count=evals_count
            )
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals

        evals = spectrum_data.energy_table.copy()
        if subtract_ground:
            evals -= evals[:, 0][:, np.newaxis]
            evals = evals[:, 1:]  # Remove ground state

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax

        if param_name == "phase":
            param_vals = param_vals / 2 / np.pi

        color = kwargs.get("color", None)
        linestyle = kwargs.get("linestyle", None)
        ax.plot(param_vals, evals[:, :evals_count], color=color, linestyle=linestyle)

        xlabel = self.PARAM_LABELS.get(param_name, param_name)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energy [GHz]")
        # ax.grid(True)

        return fig, ax

    def plot_matelem_vs_paramvals(
        self,
        operator: str,
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        select_elems: Optional[Union[int, list[tuple[int, int]]]] = None,
        mode: Literal["abs", "real", "imag", "abs_squared"] = "abs",
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
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
            Number of elements to select or list of specific elements to plot (default is [(1, 0)]).
        mode : Literal["abs", "real", "imag", "abs_squared"], optional
            Mode for plotting the matrix elements ('abs', 'real', 'imag', 'abs_squared') (default is 'abs').
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
        if select_elems is None:
            select_elems = [(1, 0)]

        # Convert select_elems to proper format early
        if isinstance(select_elems, int):
            select_elems = [
                (i, j) for i in range(select_elems) for j in range(i, select_elems)
            ]

        if spectrum_data is None:
            if param_name is None or param_vals is None:
                raise ValueError("Both param_name and param_vals must be provided.")
            evals_count = max(max(i, j) for i, j in select_elems) + 1
            spectrum_data = self.get_matelements_vs_paramvals(
                operator, param_name, param_vals, evals_count=evals_count
            )
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax

        fig.suptitle(self._generate_suptitle(param_name))
        matrixelem_table = spectrum_data.matrixelem_table[operator]

        operator_label = self.OPERATOR_LABELS.get(operator, operator)

        if param_name == "phase":
            param_vals = param_vals / 2 / np.pi

        for i, j in select_elems:
            if mode == "abs":
                values = np.abs(matrixelem_table[:, i, j])
                ylabel = rf"$|\langle {{i}} | {operator_label} | {{j}} \rangle|$"
            elif mode == "abs_squared":
                values = np.abs(matrixelem_table[:, i, j]) ** 2
                ylabel = rf"$|\langle {{i}} | {operator_label} | {{j}} \rangle|^2$"
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
        noise_channels: list[str],
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        select_elems: Optional[Union[int, list[tuple[int, int]]]] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot T1 times due to capacitive noise as a function of a parameter.

        Parameters
        ----------
        noise_channels : List[str]
            List of noise channels to plot.
        param_name : str, optional
            Name of the parameter to vary.
        param_vals : np.ndarray, optional
            Values of the parameter to vary.
        select_elems : Union[int, List[Tuple[int, int]]], optional
            Number of elements to select or list of specific elements to plot. Default is [(1, 0)].
        spectrum_data : SpectrumData, optional
            Precomputed spectral data to use.
        **kwargs
            Additional arguments for plotting.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if select_elems is None:
            select_elems = [(1, 0)]
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]

        if isinstance(select_elems, int):
            evals_count = select_elems
        elif isinstance(select_elems, list):
            evals_count = max(max(i, j) for i, j in select_elems) + 1

        if (
            spectrum_data is None
            or spectrum_data.t1_table is None
            or not all(channel in spectrum_data.t1_table for channel in noise_channels)
        ):
            spectrum_data = self.get_t1_vs_paramvals(
                noise_channels,
                param_name,
                param_vals,
                evals_count=evals_count,
                spectrum_data=spectrum_data,
            )

        param_name = spectrum_data.param_name
        param_vals = spectrum_data.param_vals

        if isinstance(select_elems, int):
            select_elems = [
                (i, j) for i in range(select_elems) for j in range(i) if i > j
            ]

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax

        rate_effective = np.zeros_like(param_vals)

        if param_name == "phase":
            param_vals = param_vals / 2 / np.pi

        for i, j in select_elems:
            for channel in noise_channels:
                t1_times = spectrum_data.t1_table[channel][:, i, j]
                rate_effective += 1 / t1_times
                ax.plot(param_vals, t1_times, label=f"{channel} ({i},{j})")

        ax.plot(
            param_vals,
            1 / rate_effective,
            label="T1 effective",
            color="black",
            linestyle="--",
        )

        xlabel = self.PARAM_LABELS.get(param_name, param_name)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$T_1$ [s]")
        ax.legend()
        # ax.grid(True)

        return fig, ax

    def plot_tphi_vs_paramvals(
        self,
        noise_channels: list[str],
        param_name: Optional[str] = None,
        param_vals: Optional[np.ndarray] = None,
        select_elems: Optional[Union[int, list[tuple[int, int]]]] = None,
        spectrum_data: Optional[SpectrumData] = None,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot Tphi times due to specified noise channels as a function of a parameter.

        Parameters
        ----------
        param_name : str
            Name of the parameter to vary.
        param_vals : np.ndarray
            Values of the parameter to vary.
        select_elems : Union[int, List[Tuple[int, int]]] optional
            The figure and axes of the plot.
        """
        if select_elems is None:
            select_elems = [(1, 0)]
        if isinstance(noise_channels, str):
            noise_channels = [noise_channels]

        if spectrum_data is None:
            operators = set()
            for channel in noise_channels:
                if channel == "flux_noise":
                    operators.add("d_hamiltonian_d_phase")
                    operators.add("d2_hamiltonian_d_phase2")
                elif channel == "charge_noise":
                    operators.add("d_hamiltonian_d_ng")
                    operators.add("d2_hamiltonian_d_ng2")
                else:
                    raise ValueError(f"Unsupported Tphi noise channel: {channel}")
            spectrum_data = self.get_matelements_vs_paramvals(
                list(operators), param_name, param_vals
            )
        else:
            param_name = spectrum_data.param_name
            param_vals = spectrum_data.param_vals

        if isinstance(select_elems, int):
            select_elems = [
                (i, j) for i in range(select_elems) for j in range(i) if i > j
            ]

        for channel in noise_channels:
            if channel not in spectrum_data.tphi_table:
                spectrum_data = self.get_tphi_vs_paramvals(
                    [channel], param_name, param_vals, spectrum_data=spectrum_data
                )

        if param_name == "phase":
            param_vals = param_vals / 2 / np.pi

        fig_ax = kwargs.get("fig_ax")
        if fig_ax is None:
            fig, ax = plt.subplots()
            fig.suptitle(self._generate_suptitle(param_name))
        else:
            fig, ax = fig_ax

        for i, j in select_elems:
            for channel in noise_channels:
                tphi_times = spectrum_data.tphi_table[channel][:, i, j]
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
            raise AttributeError(
                f"Parameter '{param_name}' does not exist in the Ferbo class."
            )
        setattr(self, param_name, val)

    def _generate_suptitle(
        self, exclude_params: Optional[Union[str, list[str]]] = None
    ) -> str:
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
                return f"{value:.4f}".rstrip("0").rstrip(".")
            return str(value)

        title_parts = [
            rf"{label} = {format_value(getattr(self, param))}"
            if param not in exclude_params
            else ""
            for param, label in self.PARAM_LABELS.items()
        ]
        return ", ".join(filter(None, title_parts))
