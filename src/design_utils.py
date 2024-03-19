import numpy as np
import scipy.constants as const
from scipy.integrate import nquad


def calculate_mutual(size:list[float],offset_position:list[float], flux_line_length:float) -> list[float]:
    '''
    TODO: Improve the documentation.
    units in micrometer! if not there will be errors of rounding.
    only valid for the moment for rectangles.
    TODO: Add for the case when you have a set of points.
    It considers the center point of the flux line as the (0,0).
    size: Size of the rectangle like (x,y)
    offset_position: The coordinate of the corner closest to the (0,0)
    flux_line_length: Length considering along the y axis.
    '''
    
    integrand = lambda x,y,l : x / np.sqrt(x**2 + (y - l)**2)**3 - x / np.sqrt(x**2 + (y + l)**2)**3

    ranges =[
        [offset_position[0], offset_position[0]+ size[0]],
        [offset_position[1], offset_position[1]+ size[1]],
        [0,flux_line_length]
    ]

    result = nquad(integrand, ranges)

    return const.mu_0/4/np.pi/2 * result[0] * 1e-6