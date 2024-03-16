import numpy as np
import scipy.constants as const
import scipy.stats
import scqubits as sq
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

phi0 = const.h/2/const.e/2/np.pi
f_temp = const.k*0.015/const.h*1e-9

Ec_area = 0.49
Ej_over_area = 48.5
El_per_junction = 35.4
C0_jja = 53e-18

class FluxoniumManager():
    def __init__(self):
        self.optimal_fluxonium = None

    def fluxonium_creator(self, params, **kwargs):
        """
        Calculate and return a Fluxonium qubit object with specified parameters.
        
        Parameters:
        - params (tuple): A tuple containing the following parameters:
            - small_jj_area (float): The area of the small Josephson junction in µm².
            - n_junctions (int): The initial number of Josephson junctions, which will be scaled by 100.
            - flux (float): The magnetic flux through the qubit, in units where the flux quantum is 1.
        - **kwargs: Additional keyword arguments to be passed to the Fluxonium constructor.
        
        Returns:
        - A Fluxonium qubit object configured with the calculated energy scales (EC, EL, EJ) and the specified flux.
        
        This function computes the hamiltonian based on experimental constrains given in params.
        """
        small_jj_area, n_junctions, flux = params

        n_junctions = n_junctions*100
        Ec_small_jj = Ec_area/small_jj_area
        Ec_JJA = const.e**2/const.h/2/(n_junctions*C0_jja)*1e-9
        Ec = scipy.stats.hmean([Ec_small_jj, Ec_JJA])
        El = El_per_junction/n_junctions
        Ej = Ej_over_area*small_jj_area
        return sq.Fluxonium(EC=Ec, EL=El, EJ=Ej, flux=flux, cutoff=50, **kwargs) #Change later to consider both fluxes.

    def _Gamma2(self,params):
        fluxonium = self.fluxonium_creator(params)
        
        Gamma2 = fluxonium.t2_effective(
                            noise_channels=['tphi_1_over_f_cc','tphi_1_over_f_flux',('t1_flux_bias_line', dict(M=1000)), 't1_inductive', ('t1_quasiparticle_tunneling', dict(Delta = 0.0002))],
                            common_noise_options=dict(T=0.015),
                            get_rate= True
                                )

        #TODO: Change with a better function to solve the problem of the anharmonicity continuiously
        if (np.abs(fluxonium.anharmonicity()) < 0.100) or (fluxonium.anharmonicity() + fluxonium.E01() < 0.100) or (fluxonium.E01() < 0.400):
            Gamma2 = 1e-6

        return Gamma2

    def minimizer(self, bounds):
        #TODO: Add the bounds as a dictionary to be more understandable.
        bounds_list = list(bounds.values())
        self.result = differential_evolution(func=self._Gamma2,bounds=bounds_list) #optimizing the T2 of the fluxonium.
        self.optimal_fluxonium = self.fluxonium_creator(self.result.x)
        self.flux_array = np.linspace(self.optimal_fluxonium.flux-0.5,self.optimal_fluxonium.flux + 0.5, 51)
    
    def plot_evals_vs_flux(self, resonator_freq, plasma_freq= None, ax = None, evals_count=6):
        if self.optimal_fluxonium is None:
            raise RuntimeError("minimizer must be run successfully before plotting.")
        
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
        if plasma_freq:
            ax.plot(self.flux_array, spec.energy_table[:,0]+plasma_freq*np.ones_like(self.flux_array), color='k', label = r'$\omega_p/2\pi$')

        ax.legend()
        ax.set_xlabel(r'$\Phi_{ext}/\Phi_0$')
        ax.set_ylabel(r'Energy (GHz)')

    def fluxonium_resonator_creator(self, resonator_frequency, EL_resonator, beta) -> sq.Circuit:
        '''
        beta: Part of the fluxonium inductance that is shared to the resonator.
        '''
        self.resonator_frequency = resonator_frequency
        EC_jja_readout = self.resonator_frequency**2/8/EL_resonator

        zp_yaml = f"""
        branches:
        - ["JJ", 1,2, {self.optimal_fluxonium.EJ}, {self.optimal_fluxonium.EC}]
        - ["L", 2,3, {self.optimal_fluxonium.EL/(1-beta)}]
        # coupling inductance
        - ["L", 1,3, {self.optimal_fluxonium.EL/beta}]
        # jja antenna readout
        - ["L", 3,4, {EL_resonator}]
        - ["C", 4,1, {EC_jja_readout}]
        """
        self.fluxonium_resonator = sq.Circuit(zp_yaml, from_file=False, ext_basis='discretized') #it works with both ext_basis.
        self.fluxonium_resonator.Φ1 = self.optimal_fluxonium.flux
        return self.fluxonium_resonator
    
    def plot_evals_fluxonium_resonator_vs_flux(self, resonator_frequency, EL_resonator, beta,evals_count=10, ax=None):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        if not hasattr(self, 'fluxonium_resonator'):
            self.fluxonium_resonator_creator(resonator_frequency, EL_resonator, beta)

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