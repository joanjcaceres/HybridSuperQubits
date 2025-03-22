import os
import h5py
import numpy as np
from qutip import Qobj
from typing import Any, Dict, List, Union, Tuple

class SpectrumData:
    def __init__(
        self,
        energy_table: np.ndarray,
        system_params: Dict[str, Any],
        param_name: str = None,
        param_vals: np.ndarray = None,
        state_table: Union[List[Qobj], np.ndarray] = None,
        matrixelem_table: Dict[str, np.ndarray] = None,
        t1_table: Dict[str, np.ndarray] = None,
        tphi_table: Dict[Tuple[int, int, str], np.ndarray] = None,
        d2E_table: Dict[str, np.ndarray] = None,
        **kwargs
    ) -> None:
        
        self.system_params = system_params
        self.param_name = param_name
        self.param_vals = param_vals
        self.energy_table = energy_table
        self.state_table = state_table if state_table is not None else []
        self.matrixelem_table = matrixelem_table if matrixelem_table is not None else {}
        self.t1_table = t1_table if t1_table is not None else {}
        self.tphi_table = tphi_table if tphi_table is not None else {}
        self.d2E_table = d2E_table if d2E_table is not None else {}
        for dataname, data in kwargs.items():
            setattr(self, dataname, data)

    def subtract_ground(self) -> None:
        """Subtract ground state energies from spectrum.
        
        The energy_table has dimensions (param_vals_count, evals_count).
        This method subtracts the ground state energy (column 0) from all states
        at each parameter value (each row) and then removes the ground state column
        which becomes identically zero.
        """
        # Reshape ground state energies to allow broadcasting
        ground_energies = self.energy_table[:, 0].reshape(-1, 1)
        self.energy_table = self.energy_table - ground_energies

    def filewrite(self, filename: str, overwrite: bool = False) -> None:
        """Save the SpectrumData to an HDF5 file."""
        
        if not overwrite and os.path.exists(filename):
            raise FileExistsError(f"The file '{filename}' already exists and overwrite is set to False.")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with h5py.File(filename, 'w') as f:
            f.create_dataset('energy_table', data=self.energy_table)
            f.create_dataset('param_vals', data=self.param_vals)
            f.create_dataset('param_name', data=np.bytes_(self.param_name))
            f.create_dataset('system_params', data=np.bytes_(str(self.system_params)))
            
            if isinstance(self.state_table, list) and isinstance(self.state_table[0], Qobj):
                state_data = np.array([state.full() for state in self.state_table])
                f.create_dataset('state_table', data=state_data)
            else:
                f.create_dataset('state_table', data=self.state_table)
                
            if self.matrixelem_table:
                for key, value in self.matrixelem_table.items():
                    f.create_dataset(f'matrixelem_table/{key}', data=value)
            if self.t1_table:
                for key, value in self.t1_table.items():
                    f.create_dataset(f't1_table/{key}', data=value)
            if self.tphi_table:
                for key, value in self.tphi_table.items():
                    f.create_dataset(f'tphi_table/{key}', data=value)

    @staticmethod
    def read(filename: str) -> 'SpectrumData':
        """Read SpectrumData from an HDF5 file."""
        with h5py.File(filename, 'r') as f:
            energy_table = f['energy_table'][:]
            param_vals = f['param_vals'][:]
            param_name = f['param_name'][()].decode('utf-8')
            system_params = eval(f['system_params'][()].decode('utf-8'))
            state_table = f['state_table'][:] if 'state_table' in f else None
            matrixelem_table = {key: f[f'matrixelem_table/{key}'][:] for key in f['matrixelem_table']} if 'matrixelem_table' in f else {}
            t1_table = {tuple(key.split(',')): f[f't1_table/{key}'][:] for key in f['t1_table']} if 't1_table' in f else {}
            tphi_table = {tuple(key.split(',')): f[f'tphi_table/{key}'][:] for key in f['tphi_table']} if 'tphi_table' in f else {}
        
        return SpectrumData(
            energy_table=energy_table,
            system_params=system_params,
            param_name=param_name,
            param_vals=param_vals,
            state_table=state_table,
            matrixelem_table=matrixelem_table,
            t1_table=t1_table,
            tphi_table=tphi_table
        )