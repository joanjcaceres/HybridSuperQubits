from typing import Any, Dict, List, Union
import numpy as np

class SpectrumData:
    def __init__(
        self,
        energy_table: np.ndarray,
        system_params: Dict[str, Any],
        param_name: str = None,
        param_vals: np.ndarray = None,
        state_table: Union[List[np.ndarray], np.ndarray] = None,
        matrixelem_table: Dict[str, np.ndarray] = None,
        t1_table: Dict[str, np.ndarray] = None,
        **kwargs
    ) -> None:
        
        self.system_params = system_params
        self.param_name = param_name
        self.param_vals = param_vals
        self.energy_table = energy_table
        self.state_table = state_table
        self.matrixelem_table = matrixelem_table if matrixelem_table is not None else {}
        self.t1_table = t1_table
        for dataname, data in kwargs.items():
            setattr(self, dataname, data)

    def subtract_ground(self) -> None:
        """Subtract ground state energies from spectrum"""
        self.energy_table -= self.energy_table[:, 0]
