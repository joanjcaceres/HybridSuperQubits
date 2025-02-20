import numpy as np

def sin_kphi_operator(k: int, dimension: int, phi_ext: float = 0) -> np.ndarray:
    """
    Generate the matrix representation of the \\sin(k\\hat{\\phi}) operator in the number basis.

    The operator is defined via the exponential representation:
        sin(k\\phi) = (e^(ik\\phi) - e^(-ik\\phi))/(2i)
    In the number basis, the matrix element corresponding to a shift by +k is 1/(2i)
    and by -k is -1/(2i). Note that 1/(2i) equals -0.5j.

    Parameters
    ----------
    k : int
        The integer multiplier of \\hat{\\phi}.
    dimension : int
        Dimension of the Hilbert space.
    phi_ext : float, optional
        External phase offset, by default 0.

    Returns
    -------
    numpy.ndarray
        Matrix representation of \\sin(k\\hat{\\phi}).

    Notes
    -----
    When k == 0, sin(0) = 0, so the operator is the zero operator.
    """
    if k == 0:
        return np.zeros((dimension, dimension))
    
    sin_kphi = np.zeros((dimension, dimension), dtype=complex)
    indices = np.arange(dimension)

    # For terms <n|exp(i k phi)|m>: m = n + k
    mask_up = indices + k < dimension
    # For terms <n|exp(-i k phi)|m>: m = n - k
    mask_down = indices - k >= 0

    # According to the definition:
    # sin(k phi) = [exp(i k phi) - exp(-i k phi)]/(2i)
    # 1/(2i) = -0.5j, so:
    sin_kphi[indices[mask_up], indices[mask_up] + k] = -0.5j * np.exp(-1j * phi_ext)
    sin_kphi[indices[mask_down], indices[mask_down] - k] = 0.5j * np.exp(1j * phi_ext)

    return sin_kphi

def cos_kphi_operator(k:int, dimension: int, phase: float = 0) -> np.ndarray:
    """
    Generate the matrix representation of the \cos(k\hat{\phi}) operator in the number basis.
    
    Parameters:
        k (int): The integer multiplier of \hat{\phi}.
        dimension (int): Dimension of the Hilbert space.
        phase (float, optional): Phase offset, by default
    
    Returns:
        numpy.ndarray: Matrix representation of \cos(k\hat{\phi}).
    """
    
    if k == 0:
        return np.eye(dimension)
    
    cos_kphi = np.zeros((dimension, dimension), dtype=np.complex128)
    indices = np.arange(dimension)
    
    mask_up = indices + k < dimension
    mask_down = indices - k >= 0
    
    cos_kphi[indices[mask_up], indices[mask_up] + k] = 0.5 * np.exp( - 1j * phase)
    cos_kphi[indices[mask_down], indices[mask_down] - k] = 0.5 * np.exp(1j * phase)
    
    return cos_kphi