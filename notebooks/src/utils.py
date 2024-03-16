import numpy as np
import scipy.constants as const
import scipy.stats
import scqubits as sq
from scipy.optimize import differential_evolution

phi0 = const.h/2/const.e/2/np.pi
f_temp = const.k*0.015/const.h*1e-9

Ec_area = 0.49
Ej_over_area = 48.5
El_per_junction = 35.4
C0_jja = 53e-18

def fluxo(params, **kwargs):
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
    small_jj_area, n_junctions,flux = params

    n_junctions = n_junctions*100
    Ec_small_jj = Ec_area/small_jj_area
    Ec_JJA = const.e**2/const.h/2/(n_junctions*C0_jja)*1e-9
    Ec = scipy.stats.hmean([Ec_small_jj, Ec_JJA])
    El = El_per_junction/n_junctions
    Ej = Ej_over_area*small_jj_area
    return sq.Fluxonium(EC=Ec, EL=El, EJ=Ej, flux=flux, cutoff=120, **kwargs)

def optimizer(params):
    fluxonium = fluxo(params)
    
    Gamma2 = 1/fluxonium.t2_effective(
                        noise_channels=['tphi_1_over_f_cc','tphi_1_over_f_flux',('t1_flux_bias_line', dict(M=1000)), 't1_inductive', ('t1_quasiparticle_tunneling', dict(Delta = 0.0002))],
                        common_noise_options=dict(T=0.015)
                            )

    #TODO: Change with a better function to solve the problem of the anharmonicity continuiously

    if np.abs(fluxonium.anharmonicity()) or fluxonium.anharmonicity()+fluxonium.E01 < 0.100:
        Gamma2 = Gamma2*1e3

    return Gamma2

def optimal_fluxonium(bounds):
    #TODO: Add the bounds as a dictionary to be more understandable.
    bounds_list = list(bounds.values())
    result = differential_evolution(func=optimizer,bounds=bounds_list) #optimizing the T2 of the fluxonium.
    print(result)
    return fluxo(result.x)

# def calculation_C_JJA(C0, Cj, n):
#     n = int(n)
#     if n == 0:
#         return Cj
#     else:
#         previous_result = calculation_C_JJA(C0, Cj, n - 1)
#         return scipy.stats.hmean([C0 + previous_result, C0])