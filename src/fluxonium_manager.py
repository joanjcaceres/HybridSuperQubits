import numpy as np
import scipy.constants as const
from src.fluxonium import calculate_CQPS_rate
from src.utilities import *
import scqubits as sq
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

phi0 = const.h/2/const.e/2/np.pi
Rq = const.h/(2*const.e)**2
f_temp = const.k*0.015/const.h*1e-9

totalIslandsLength = 68 #µm
fluxLineCapacitance = 3.8 #fF #It can be zero if we use a coil instead of a flux line.
mutualInductance = 1000 # \Phi_0/A

with open("../data/junctions.yaml", 'r') as file:
    junctions_data = yaml.safe_load(file)

CjperArea = junctions_data['CJ_per_area']
LjArea = junctions_data['LJ_times_area']
C0perJunction = junctions_data['C0_per_junction']
C0perLength = junctions_data['C0_per_length']

junctionArea = junctions_data['junction_width'] * junctions_data['junction_length']

Ec_area = C_to_Ec(CjperArea * 1e-15) * 1e-9 #Area in µm^2.
Ej_over_area = L_to_El(LjArea * 1e-9) * 1e-9 #Area in µm^2.

El_times_junction = Ej_over_area * junctionArea
Ec_junction = Ec_area / junctionArea
Ec0_per_junction = C_to_Ec(C0perJunction / 12 * 1e-18) * 1e-9

# C0_jja = 53e-18
# Lj_per_junction = 4.62e-9
# Z_c0 = np.sqrt(Lj_per_junction/C0_jja)

# Improve the optimizer in order to take the junction area of the array of as a parameter too.

class FluxoniumResult:
    def __init__(self, fluxonium: sq.Fluxonium, params: list, gamma_inverse: float):
        self.fluxonium: sq.Fluxonium = fluxonium
        self.params: list = params
        self.gamma_inverse: float = gamma_inverse

    def __repr__(self):
        return (f"FluxoniumResult(fluxonium={self.fluxonium}, "
                f"params = {self.params}, "
                f"T2 = {round(self.gamma_inverse*1e-6,4)} ms)")

class FluxoniumManager():
    def __init__(self):
        self.results = {
            "flux_0": None,
            "flux_0_5": None
        }
        self.original_bounds = None


    def fluxonium_creator(self, params, flux, **kwargs):
        """
        Creates and returns a Fluxonium qubit object with specified parameters.

        Parameters
        ----------
        params : tuple
            A tuple containing the following parameters:
            - small_jj_area (float): The area of the small Josephson junction in µm².
            - n_junctions (int): The number of Josephson junctions.
        flux : float
            The magnetic flux through the qubit, in units where the flux quantum is 1.
        **kwargs : dict
            Additional keyword arguments to be passed to the Fluxonium constructor.

        Returns
        -------
        sq.Fluxonium
            A Fluxonium qubit object configured with the calculated energy scales 
            (EC, EL, EJ) and the specified flux.
        """
        small_jj_area, n_junctions = params

        # n_junctions *= 100 #TODO: Change the scaling in the optimizator, not here.
        
        C0_islands = (C0perJunction * n_junctions + 2 * C0perLength * totalIslandsLength) / 12 #fF 
        Cj_array = CjperArea * junctionArea / n_junctions #fF
        Cj_small_jj = CjperArea * small_jj_area #fF
        
        EC = C_to_Ec((C0_islands + Cj_array + Cj_small_jj + fluxLineCapacitance) * 1e-15) * 1e-9 # GHz
        EL = L_to_El(LjArea / junctionArea * n_junctions * 1e-9) * 1e-9 # GHz
        EJ = L_to_El(LjArea / small_jj_area * 1e-9) * 1e-9 # GHz
        
        # return sq.Fluxonium(EC=EC, EL=EL, EJ=EJ, flux=flux, cutoff=50, **kwargs) #Change later to consider both fluxes.
        return sq.Fluxonium(EC=EC, EL=EL, EJ=EJ, flux=flux, cutoff=50, **kwargs) #Change later to consider both fluxes.

    def _Gamma2(self,params_normalized, flux):
        """
        Calculates the decoherence rate Gamma2 for a given Fluxonium qubit.

        Parameters
        ----------
        params_normalized : tuple
            Normalized parameters for the Fluxonium qubit.
        flux : float
            The magnetic flux through the qubit.

        Returns
        -------
        float
            The calculated decoherence rate Gamma.
        """
        # Denormalize parameters
        params = self._denormalize_params(params_normalized)
        
        fluxonium = self.fluxonium_creator(params, flux)
        
        Gamma2 = fluxonium.t2_effective(
                            noise_channels=[
                                'tphi_1_over_f_cc','tphi_1_over_f_flux',
                                ('t1_flux_bias_line', dict(M=mutualInductance)),
                                't1_inductive',
                                ('t1_quasiparticle_tunneling', dict(Delta = 0.0002))
                                ],
                            common_noise_options=dict(T=0.015),
                            get_rate= True,
                                )
        small_jj_area, n_junctions = params 
        # n_junctions *= 100 #TODO: Change this way of normalizing the amount of junctions
        
        GammaCQPS = calculate_CQPS_rate(
            fluxonium=fluxonium,
            ECj= Ec_junction,
            EJj= El_times_junction,
            n_junctions=n_junctions,
            )
        
        Gamma = np.sqrt(Gamma2**2 + GammaCQPS**2) # Based on arXiv:2404.02989v1 p.6

        #TODO: Change with a better function to solve the problem of the anharmonicity continuiously
        if (np.abs(fluxonium.anharmonicity()) < 0.100) or \
           (fluxonium.anharmonicity() + fluxonium.E01() < 0.100) or \
           (fluxonium.E01() < 0.400):
            Gamma = 1e-6

        return Gamma

    def _normalize_bounds(self, bounds):
        """
        Normalizes the bounds to the range [0, 1] for optimization.

        Parameters
        ----------
        bounds : list of tuples
            Each tuple contains (min, max) bounds for the corresponding parameter.

        Returns
        -------
        list of tuples
            Normalized bounds where each tuple is (0, 1).
        """
        return [(0, 1) for _ in bounds]
    
    def _denormalize_params(self, params_normalized):
        """
        Denormalizes parameters from the normalized range [0, 1] back to the original scale
        using the provided bounds.

        Parameters
        ----------
        params_normalized : list
            Parameters normalized in the range [0, 1].
        bounds : list of tuples
            Each tuple contains (min, max) bounds for the corresponding parameter.

        Returns
        -------
        list
            Denormalized parameters in their original scale.
        """
        return [low + param * (high - low) for param, (low, high) in zip(params_normalized, self.original_bounds)]
    
    def minimizer(self, bounds:dict):
        """
        Optimizes the parameters of a Fluxonium qubit to find the minimum decoherence rate.

        Parameters
        ----------
        bounds : dict
            Dictionary with the bounds of the parameters to be optimized.

        Returns
        -------
        None
            The results of the optimization are stored in the class attributes.
        """
        #TODO: Add the bounds as a dictionary to be more understandable.
        # Store the original bounds
        self.original_bounds = list(bounds.values())
        
        # Set normalized bounds in [0, 1] for optimization
        bounds_normalized = self._normalize_bounds(self.original_bounds)
        
        # Optimization at flux=0
        result_flux_0 = differential_evolution(
            func=lambda params: self._Gamma2(params, flux=0), 
            bounds=bounds_normalized
        )
        gamma_flux_0 = self._Gamma2(result_flux_0.x, flux=0)
        params_flux_0 = self._denormalize_params(result_flux_0.x)
        gamma_inverse_flux_0 = 1 / gamma_flux_0
        fluxonium_flux_0 = self.fluxonium_creator(params_flux_0, flux=0)

        # Store results for flux 0
        self.results["flux_0"] = FluxoniumResult(
            fluxonium=fluxonium_flux_0,
            params=params_flux_0,
            gamma_inverse=gamma_inverse_flux_0
        )

        # Optimization at flux = 0.5
        result_flux_0_5 = differential_evolution(
            func=lambda params: self._Gamma2(params, flux=0.5), 
            bounds=bounds_normalized
        )
        gamma_flux_0_5 = self._Gamma2(result_flux_0_5.x, flux=0.5)
        params_flux_0_5 = self._denormalize_params(result_flux_0_5.x)
        gamma_inverse_flux_0_5 = 1 / gamma_flux_0_5
        fluxonium_flux_0_5 = self.fluxonium_creator(params_flux_0_5, flux=0.5)

        # Store results for flux 0.5
        self.results["flux_0_5"] = FluxoniumResult(
            fluxonium=fluxonium_flux_0_5,
            params=params_flux_0_5,
            gamma_inverse=gamma_inverse_flux_0_5
        )
        
    def get_optimization_results(self):
        """
        Returns the results of the optimization, including the optimal Fluxonium instances,
        the optimized parameters, and the inverse of Gamma for both flux=0 and flux=0.5.

        Returns
        -------
        dict
            A dictionary with the results for both flux=0 and flux=0.5.
        """
        return self.results
    
    def plot_evals_vs_flux(self, resonator_freq, ax = None, evals_count=6):
        # if self.optimal_fluxonium is None:
        #     raise RuntimeError("minimizer must be run successfully before plotting.")
        
        fluxonium = self.optimal_fluxonium
        if ax is None:
            fig,ax = plt.subplots(1,1)
            fig.suptitle(f'Ec: {np.round(fluxonium.EC,3)}, El: {np.round(fluxonium.EL,3)}, Ej: {np.round(fluxonium.EJ,3)}')


        spec = fluxonium.get_spectrum_vs_paramvals(param_name='flux',param_vals=self.flux_array, evals_count=evals_count, subtract_ground=True)
        self.evals_fluxonium_vs_flux = spec.energy_table

        ax.plot(self.flux_array,spec.energy_table)
        ax.plot(self.flux_array, spec.energy_table[:,0]+resonator_freq*np.ones_like(self.flux_array), color='k', linestyle='--', label = r'$|1g\rangle$')
        ax.plot(self.flux_array, spec.energy_table[:,1]+resonator_freq*np.ones_like(self.flux_array), color='k',linestyle='dotted', label = r'$|1e\rangle$')
        ax.plot(self.flux_array, spec.energy_table[:,0]+f_temp*np.ones_like(self.flux_array), color='red',linestyle='dotted', label = r'$f_{temp}$')
  
        ax.legend()
        ax.set_xlabel(r'$\Phi_{ext}/\Phi_0$')
        ax.set_ylabel(r'Energy (GHz)')

    def fluxonium_resonator_creator(self, resonator_frequency, beta) -> sq.Circuit:
        '''
        beta: Part of the fluxonium inductance that is shared to the resonator.
        Notes:
            Only valid by the moment when you have a JJA based readout resonator.
        '''
        self.resonator_frequency = resonator_frequency
        EL_jja_resonator = resonator_frequency*Rq/Z_c0/2/np.pi**2 #Only valid when N >> pi
        EC_jja_readout = self.resonator_frequency**2/8/EL_jja_resonator
        EL_shared = self.optimal_fluxonium.EL/beta
        EL_only_resonator = 1/(1/EL_jja_resonator - 1/EL_shared)

        zp_yaml = f"""
        branches:
        - ["JJ", 1,2, {self.optimal_fluxonium.EJ}, {self.optimal_fluxonium.EC}]
        - ["L", 2,3, {self.optimal_fluxonium.EL/(1-beta)}]
        # coupling inductance
        - ["L", 1,3, {EL_shared}]
        # jja antenna readout
        - ["L", 3,4, {EL_only_resonator}]
        - ["C", 4,1, {EC_jja_readout}]
        """
        self.fluxonium_resonator = sq.Circuit(zp_yaml, from_file=False, ext_basis='discretized') #it works with both ext_basis.
        self.fluxonium_resonator.Φ1 = self.optimal_fluxonium.flux
        return self.fluxonium_resonator
    
    def plot_evals_fluxonium_resonator_vs_flux(self, resonator_frequency, beta,evals_count=10, ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        self.fluxonium_resonator_creator(resonator_frequency, beta)

        spec = self.fluxonium_resonator.get_spectrum_vs_paramvals(param_name='Φ1',param_vals=self.flux_array,evals_count=evals_count, subtract_ground=True)
        self.evals_fluxonium_resonator_vs_flux = spec.energy_table
        ax.plot(self.flux_array,self.evals_fluxonium_resonator_vs_flux)
        ax.set_xlabel(r'$\Phi_{ext}/\Phi_0$')
        ax.set_ylabel(r'Energy (GHz)')


# def calculation_C_JJA(C0, Cj, n):
#     n = int(n)
#     if n == 0:
#         return Cj
#     else:
#         previous_result = calculation_C_JJA(C0, Cj, n - 1)
#         return scipy.stats.hmean([C0 + previous_result, C0])