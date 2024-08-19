import os
import re
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.integrate import nquad
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import h5py
import yaml
import inspect


def extract_hdf5_data(file_path):
    data = {}

    def recursive_extraction(name, obj):
        if isinstance(obj, h5py.Dataset):
            # Manejar datasets escalares
            if obj.shape == ():
                data[name] = obj[()]
            else:
                data[name] = obj[:]
        elif isinstance(obj, h5py.Group):
            data[name] = {'_attrs': dict(obj.attrs)}
            for key, val in obj.items():
                recursive_extraction(f"{name}/{key}", val)

    with h5py.File(file_path, 'r') as f:
        # Extraer atributos globales
        data['_global_attrs'] = dict(f.attrs)
        
        # Extraer datasets y atributos de grupos
        for key, val in f.items():
            recursive_extraction(key, val)

    return data

def load_data(base_file_path):
    txt_file_path = base_file_path + '.txt'
    hdf5_file_path = base_file_path + '.hdf'
    
    if os.path.exists(txt_file_path):
        return load_txt_data(base_file_path)
    elif os.path.exists(hdf5_file_path):
        return load_hdf5_data(base_file_path)
    else:
        raise FileNotFoundError(f"Neither {txt_file_path} nor {hdf5_file_path} exist.")


def load_txt_data(base_name):
    """
    Load data from a base name.

    Parameters:
    base_name (str): Base name of the files and directory.

    Returns:
    dict: Dictionary containing all data columns.
    """
    # Define paths to the files and directories based on the base name
    x_file = f'{base_name}.txt'
    y_directory = f'{base_name}-children'

    # Initialize the dictionary
    data_dict = {}

    # Initialize the mask variable
    mask = []

    # Load X-axis data
    with open(x_file, 'r') as file:
        lines = file.readlines()

    # Extract parameters from the header`
    parameters = read_parameters(lines)
    if parameters:
        data_dict['parameters'] = parameters

    # Find the end of the header and read column headers
    header_end_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == '#end of header':
            header_end_idx = i
            break

    # Read the column headers
    column_headers = lines[header_end_idx + 1].strip().split()

    # Read the data after the header
    data = [line.strip().split() for line in lines[header_end_idx + 2:] if line.strip()]
    data = np.array(data, dtype=float)

    num_rows = data.shape[0]

    # Populate the data dictionary with columns from x_file
    for i, header in enumerate(column_headers):
        data_dict[header] = data[:, i]

    # Check if the directory exists
    if os.path.isdir(y_directory):
        filenames = sorted([f for f in os.listdir(y_directory) if f.endswith('.txt')])

        # mask = []

        for idx, filename in enumerate(filenames):
            if idx >= num_rows:
                break
            y_file = os.path.join(y_directory, filename)
            with open(y_file, 'r') as file:
                lines = file.readlines()

            # Find the end of the header
            header_end_idx = 0
            for i, line in enumerate(lines):
                if line.strip() == '#end of header':
                    header_end_idx = i
                    break

            # Read the column headers
            file_column_headers = lines[header_end_idx + 1].strip().split()

            # Read the data after the header
            data = [line.strip().split() for line in lines[header_end_idx + 2:] if line.strip()]
            data = np.array(data, dtype=float)
            if idx == 0:
                correct_shape = data.shape
            if not data.shape == correct_shape:
                mask.append(idx)

            # Populate the data dictionary with columns from y_directory
            for i, header in enumerate(file_column_headers):
                if header not in data_dict:
                    data_dict[header] = []
                if data.shape == correct_shape:
                    data_dict[header].append(data[:, i])

        for key in data_dict:
            if isinstance(data_dict[key], list):
                data_dict[key] = np.array(data_dict[key])

    for i, header in enumerate(column_headers):
        if isinstance(data_dict[header], np.ndarray):
            data_dict[header] = np.delete(data_dict[header], mask, axis=0)

    return data_dict

def load_hdf5_data(file_path):
    """
    Valid to obtain the data saved in HDF format in the Quantrolab's Datacube.
    """
    hdf5_file_path = f"{file_path}.hdf"
    with h5py.File(hdf5_file_path, 'r') as f:
        # Carga los parámetros y meta atributos
        parameters = yaml.safe_load(filter_lines(f.attrs['parameters']))
        meta = yaml.safe_load(filter_lines(f.attrs['meta']))
        
        # Extrae el fieldMap del meta principal
        main_field_map = meta.get('fieldMap', {})
        
        # Inicializa el diccionario de datos con los parámetros
        data_dict = {'parameters': parameters}
        
        # Si existe un table en el grupo principal, procesarlo
        if 'table' in f:
            main_table = f['table']
            for field, index in main_field_map.items():
                data_dict[field] = []
                for row_idx in range(main_table.shape[0]):
                    data_dict[field].append(main_table[row_idx, index])
        
        # Ordena los nombres de los subgrupos numéricamente
        subgroups_sorted = sorted(f['children'].keys(), key=lambda x: int(x))
        for idx, group in enumerate(subgroups_sorted):
            table = f['children'][group]['table']
            if idx == 0:
                # Obtiene el meta atributo del primer subgrupo para el fieldMap del subgrupo
                meta_child = yaml.safe_load(filter_lines(f['children'][group].attrs['meta']))
                child_field_map = meta_child['fieldMap']
                
                # Inicializa las listas para cada campo del fieldMap del subgrupo
                for field in child_field_map:
                    if field not in data_dict:
                        data_dict[field] = [[] for _ in range(table.shape[0])]
                    
            # Extrae los datos del dataset y los agrega a las listas correspondientes
            for field, index in child_field_map.items():
                # Agrega los datos a la estructura correcta
                for row_idx in range(table.shape[0]):
                    data_dict[field][row_idx].append(table[row_idx, index])
        
        # Convertir las listas a arrays numpy y asegurarse de que la forma sea correcta
        for field in data_dict:
            if field != 'parameters':
                data_dict[field] = np.array(data_dict[field]).T
            
    return data_dict

# def load_hdf5_data(file_path):
    hdf5_file_path = f"{file_path}.hdf"
    with h5py.File(hdf5_file_path, 'r') as f:
        # Carga los parámetros y meta atributos
        parameters = yaml.safe_load(filter_lines(f.attrs['parameters']))
        meta = yaml.safe_load(filter_lines(f.attrs['meta']))
        
        # Extrae el fieldMap del meta principal
        main_field_map = meta.get('fieldMap', {})
        
        # Inicializa el diccionario de datos con los parámetros
        data_dict = {'parameters': parameters}
        
        # Si existe un table en el grupo principal, procesarlo
        if 'table' in f:
            main_table = f['table']
            for field, index in main_field_map.items():
                data_dict[field] = []
                for row_idx in range(main_table.shape[0]):
                    data_dict[field].append(main_table[row_idx, index])
        
        # Ordena los nombres de los subgrupos numéricamente
        subgroups_sorted = sorted(f['children'].keys(), key=lambda x: int(x))
        for idx, group in enumerate(subgroups_sorted):
            table = f['children'][group]['table']
            if idx == 0:
                # Obtiene el meta atributo del primer subgrupo para el fieldMap del subgrupo
                meta_child = yaml.safe_load(filter_lines(f['children'][group].attrs['meta']))
                child_field_map = meta_child['fieldMap']
                
                # Inicializa las listas para cada campo del fieldMap del subgrupo
                for field in child_field_map:
                    if field not in data_dict:
                        data_dict[field] = []
                    
            # Extrae los datos del dataset y los agrega a las listas correspondientes
            for field, index in child_field_map.items():
                if len(data_dict[field]) == 0:
                    # Inicializa la lista de listas si es la primera vez
                    data_dict[field] = [[] for _ in range(table.shape[0])]
                # Agrega los datos a la estructura correcta
                for row_idx in range(table.shape[0]):
                    data_dict[field][row_idx].append(table[row_idx, index])
        
        # Convertir las listas a arrays numpy y asegurarse de que la forma sea correcta
        for field in data_dict:
            if field != 'parameters':
                data_dict[field] = np.array(data_dict[field])
            
    return data_dict

def filter_lines(yaml_str):
    filtered_lines = []
    for line in yaml_str.split('\n'):
        if '!!python/' not in line:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)

def calculate_mutual(loop_size:list[float],offset_position:list[float], flux_line_length:float, flux_line_width:float) -> float:
    """
    Calculates the mutual inductance between a rectangular loop and a triangular flux line.

    The calculation assumes units are in micrometers to avoid rounding errors. Currently, the function is 
    only valid for rectangular loops and assumes a triangular shape of the flux line, with the apex of the 
    triangle located at the origin (0,0) of the coordinate system.

    Parameters:
    - loop_size: A list of two floats representing the size of the rectangle in micrometers. The format is [width, height].
    - offset_position: A list of two floats indicating the coordinates (in micrometers) of the rectangle's corner 
      closest to the origin (0,0). The format is [x_offset, y_offset].
    - flux_line_length: The length of one side of the triangular flux line along the y-axis, in micrometers.
    - flux_line_width: The width of the base of the triangular flux line, in micrometers.

    Returns:
    - The calculated mutual inductance in Henrys.

    Note:
    - The function is currently only implemented for rectangular loops and triangular flux lines.
    - Future improvements might include support for arbitrary shapes defined by a set of points.

    The mutual inductance is calculated by integrating over the specified rectangle and flux line dimensions,
    considering the magnetic field generated by the triangular flux line. The center point of the flux line 
    is considered to be at (0,0).
    """
    
    integrand = lambda x,y,l,w : (x-w) / np.sqrt((x-w)**2 + (y - l)**2)**3 - (x-w) / np.sqrt((x-w)**2 + (y + l)**2)**3

    ranges =[
        [offset_position[0], offset_position[0]+ loop_size[0]],
        [offset_position[1], offset_position[1]+ loop_size[1]],
        [0,flux_line_length],
        [-flux_line_width/2,flux_line_width/2]
    ]

    result = nquad(integrand, ranges)

    return const.mu_0/4/np.pi/2 * result[0]/flux_line_width * 1e-6

def load_data_and_metadata(file_path):
    datasets = {}
    metadata = {}

    def load_group(group, path=''):
        # Load attributes as metadata
        for attr in group.attrs:
            metadata_key = f"{path}/{attr}" if path else attr
            metadata[metadata_key] = group.attrs[attr]

        # Load datasets or dive into subgroups
        for name in group:
            item_path = f"{path}/{name}" if path else name
            if isinstance(group[name], h5py.Dataset):
                datasets[item_path] = group[name][...]
            elif isinstance(group[name], h5py.Group):
                load_group(group[name], item_path)

    with h5py.File(file_path, 'r') as file:
        load_group(file)

    return datasets, metadata

def plot2D(data_dict, x_key, y_key, color_key, title=None, plot_style='line', fig=None, ax=None, **kwargs):
    """
    Plot multiple 2D lines or points with colors changing gradually based on a given array of values.

    Parameters:
    data_dict (dict): Dictionary containing the data arrays.
    x_key (str): Key for the x_values to plot.
    y_key (str): Key for the y_values to plot.
    color_key (str): Key for the values used to determine the color of each line or point.
    title (str, optional): Title of the plot.
    plot_style (str, optional): Style of plot ('line', 'point', 'both'). Default is 'line'.
    fig (matplotlib.figure.Figure, optional): Figure object to use for plotting.
    ax (matplotlib.axes.Axes, optional): Axes object to use for plotting.
    kwargs: Additional keyword arguments to pass to the plot function.
    """
    if fig is None or ax is None:
        figsize = kwargs.pop('figsize', (8, 6))
        fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis if not provided
    
    x_data = data_dict[x_key]
    y_data = data_dict[y_key]
    color_values = data_dict[color_key]
    
    # Check dimensions
    assert x_data.shape == y_data.shape, "x_data and y_data must have the same shape"
    assert x_data.shape[0] == color_values.shape[0], "Number of rows in x_data must match length of color_values"

    # Create a colormap
    cmap = plt.get_cmap('viridis')  # You can choose another colormap like 'plasma', 'inferno', etc.
    norm = plt.Normalize(vmin=color_values.min(), vmax=color_values.max())
    
    lc = None  # Initialize lc to None for checking later
    sc = None  # Initialize sc to None for checking later

    if plot_style in ['line', 'both']:
        # Create segments for LineCollection
        segments = [np.column_stack([x_data[i], y_data[i]]) for i in range(y_data.shape[0])]
        lc = LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
        lc.set_array(color_values)
        # Add the LineCollection to the plot
        ax.add_collection(lc)

    if plot_style in ['point', 'both']:
        # Scatter plot for points
        for i in range(y_data.shape[0]):
            sc = ax.scatter(x_data[i], y_data[i], c=np.full_like(x_data[i], fill_value=color_values[i]), cmap=cmap, norm=norm, **kwargs)

    # Adjust plot limits
    ax.set_xlim(x_data.min(), x_data.max())
    ax.set_ylim(y_data.min(), y_data.max())

    # Add labels and title
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    if title is not None:
        ax.set_title(title)

    # Add colorbar
    if lc is not None:
        cbar = fig.colorbar(lc, ax=ax)
    elif sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(color_key)
    
    fig.tight_layout()

    return fig, ax

def plot3D(data_dict, x_key, y_key, z_key, title=None, flatten_horizontal=False, flatten_vertical=False, fig=None, ax=None, add_colorbar=True, **kwargs):
    """
    Plot a 2D phase map using the provided X-axis values, Y-axis values, and Z-axis matrix.

    Parameters:
    data_dict (tuples): Tuples containing (x_values, y_values, Z_values).
    x_key (str): Key for the x_values to plot.
    y_key (str): Key for the y_values to plot.
    z_key (str): Key for the z_values to plot.
    title (str): Title of the plot.
    correct_horizontal (bool): Whether to apply horizontal correction.
    correct_vertical (bool): Whether to apply vertical correction.
    fig (matplotlib.figure.Figure, optional): Figure object to use for plotting.
    ax (matplotlib.axes.Axes, optional): Axes object to use for plotting.
    kwargs: Additional keyword arguments to pass to pcolormesh.
    """
    if fig is None or ax is None:
        figsize= kwargs.pop('figsize', (8, 6))
        fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis if not provided
    
    x_data = data_dict[x_key]
    y_data = data_dict[y_key]
    z_data = data_dict[z_key]

    # Apply corrections if specified
    if x_data.ndim == 1:
        Y = y_data
        X = np.tile(x_data, (Y.shape[1], 1)).T
        if flatten_horizontal:
            z_data = horizontal_flatten(z_data)
        if flatten_vertical:
            z_data = vertical_flatten(z_data)
    elif y_data.ndim == 1: # In case it's inverted the axis.
        X = x_data
        Y = np.tile(y_data, (X.shape[1], 1)).T
        if flatten_vertical:
            z_data = horizontal_flatten(z_data)
        if flatten_horizontal:
            z_data = vertical_flatten(z_data)

    # Plot using pcolormesh
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    mesh = ax.pcolormesh(X, Y, z_data, shading='auto', vmin=vmin, vmax=vmax,**kwargs)
    
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_key)
    # ax.set_ylabel(y_key)
    if add_colorbar:
        cbar = fig.colorbar(mesh, ax=ax, label=z_key)
    fig.tight_layout()
    
    return fig, ax, mesh
    
def plot3Ds(data_dicts, x_key, y_key, z_key, title=None, flatten_horizontal=False, flatten_vertical=False, fig=None, ax=None,**kwargs):
    if fig is None or ax is None:
        figsize = kwargs.pop('figsize', (8, 6))
        fig, ax = plt.subplots(figsize=figsize)
    
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    
    # Check that all data_dicts have the required keys and z_key contains arrays
    for key, data_dict in data_dicts.items():
        if z_key not in data_dict or not isinstance(data_dict[z_key], np.ndarray):
            raise ValueError(f"Each data_dict must contain the key '{z_key}' with a numpy array as its value.")
    
    if vmin is None or vmax is None:
        z_data_combined = np.concatenate([data_dict[z_key].flatten() for key, data_dict in data_dicts.items()])
        vmin = vmin if vmin is not None else np.min(z_data_combined)
        vmax = vmax if vmax is not None else np.max(z_data_combined)
        
    last_mesh = None
    
    for key, data_dict in data_dicts.items():
        fig, ax, last_mesh = plot3D(data_dict, x_key, y_key, z_key, title='', flatten_horizontal=flatten_horizontal, flatten_vertical=flatten_vertical, fig=fig, ax=ax, vmin=vmin, vmax=vmax, add_colorbar=False, **kwargs)
        
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=ax, orientation='vertical')
        cbar.ax.set_ylabel(z_key)
    
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    
    return fig,ax

def vertical_flatten(phase_matrix):
    """
    Calculate the vertical average and subtract it from each row in the phase matrix.

    Parameters:
    phase_matrix (np.ndarray): Matrix of phase values.

    Returns:
    np.ndarray: Phase matrix with vertical averages subtracted.
    """
    vertical_mean = np.nanmean(phase_matrix, axis=1)  # Calculate mean of each row
    phase_matrix_corrected = phase_matrix - vertical_mean[:, np.newaxis]  # Subtract the mean from each row
    return phase_matrix_corrected

def horizontal_flatten(phase_matrix):
    """
    Calculate the horizontal average and subtract it from each column in the phase matrix.

    Parameters:
    phase_matrix (np.ndarray): Matrix of phase values.

    Returns:
    np.ndarray: Phase matrix with horizontal averages subtracted.
    """
    horizontal_mean = np.nanmean(phase_matrix, axis=0)  # Calculate mean of each column
    phase_matrix_corrected = phase_matrix - horizontal_mean  # Subtract the mean from each column
    return phase_matrix_corrected

def plot_1d_line(datasets, x_key, y_key, z_key, fixed_x=None, fixed_y=None, tolerance=0.01, apply_vertical_correction=False, apply_horizontal_correction=False, 
                 fig=None, ax=None, **kwargs):
    """
    Plot a 1D line at a fixed Y value on an existing phase map.

    Parameters:
    datasets (list of dict): List of dictionaries containing all data columns.
    fixed_y (float): Fixed Y value to plot a 1D line.
    x_key (str): Key for the x_values to plot.
    y_key (str): Key for the y_values to plot.
    z_key (str): Key for the z_values to plot.
    tolerance (float): Tolerance within which the nearest Y value must be to fixed_y.
    apply_vertical_correction (bool): Whether to apply vertical correction to the z_values.
    apply_horizontal_correction (bool): Whether to apply horizontal correction to the z_values.
    fig (matplotlib.figure.Figure, optional): Figure object to use for plotting.
    ax (matplotlib.axes.Axes, optional): Axes object to use for plotting.
    kwargs: Additional keyword arguments to pass to the plot function.
    """
    if fig is None or ax is None:
        figsize =kwargs.pop('figsize', (8,6))
        fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis if not provided
    
    if fixed_x is not None and fixed_y is not None:
        raise ValueError("Both fixed_x and fixed_y shouldn't be different to None.")
    
    x_data = datasets[x_key]
    y_data = datasets[y_key]
    z_data = datasets[z_key]

    if x_data.ndim == 1:
        Y = y_data
        X = np.tile(x_data, (Y.shape[1], 1)).T
        if apply_horizontal_correction:
            z_data = horizontal_flatten(z_data)
        if apply_vertical_correction:
            z_data = vertical_flatten(z_data)
    elif y_data.ndim == 1: # In case it's inverted the axis.
        X = x_data
        Y = np.tile(y_data, (X.shape[1], 1)).T
        if apply_vertical_correction:
            z_data = horizontal_flatten(z_data)
        if apply_horizontal_correction:
            z_data = vertical_flatten(z_data)

    #This only works when x_data is 1D array.

    if fixed_y is not None:
        y_index = (np.abs(Y[:,0] - fixed_y)).argmin()
        z_1d = z_data[y_index,:]

        y_nearest = Y[y_index,0]
        if np.abs(y_nearest - fixed_y) > tolerance:
            print(f"Skipping dataset with y_nearest = {y_nearest:.2f} (outside tolerance of {tolerance})")


        ax.plot(x_data[0,:], z_1d, label=f'Y = {y_nearest:.2f}', **kwargs)
        ax.set_xlabel(x_key)
        ax.set_ylabel(z_key)
    elif fixed_x is not None:
        x_index = (np.abs(X - fixed_x)).argmin()
        z_1d = z_data[x_index, :]

        x_nearest = X[x_index]
        if np.abs(x_nearest - fixed_x) > tolerance:
            print(f"Skipping dataset with x_nearest = {x_nearest:.2f} (outside tolerance of {tolerance})")

        ax.plot(z_1d,y_data[0,:], label=f'X = {x_nearest:.2f}', **kwargs)
    ax.set_xlabel(z_key)
    ax.set_ylabel(y_key)

    ax.legend()
    fig.tight_layout()
    return fig, ax

def read_parameters(lines):
    in_parameters_section = False
    parameters = {}

    for line in lines:
        stripped_line = line.strip()
        # Detectar la línea que contiene 'parameters:'
        if stripped_line.startswith('parameters:'):
            in_parameters_section = True
        elif stripped_line == '#end of header':
            break
        
        # Procesar solo las líneas dentro de la sección 'parameters:'
        if in_parameters_section:
            if ':' in stripped_line:
                key_value_pair = stripped_line.split(': ', 1)
                if len(key_value_pair) == 2:
                    key, value = key_value_pair
                    # Remover posibles comas y espacios al final
                    value = value.rstrip(', ')
                    # Intentar manejar valores que sean numéricos o listas
                    try:
                        if '[' in value and ']' in value:  # Maneja valores tipo lista
                            value = eval(value)  # Evalúa de manera segura la cadena a una lista de Python
                        else:
                            value = float(value)
                    except ValueError:
                        pass  # Dejar el valor como está si no es un número o lista
                    
                    parameters[key] = value

    return parameters if parameters else None

def last_measurement(directory):
    max_number = None
    # Regex to match the file pattern
    file_pattern = re.compile(r'FBW10S4_(\d+)\.txt')

    for filename in os.listdir(directory):
        # Check if the filename matches the pattern
        match = file_pattern.match(filename)
        if match:
            number = int(match.group(1))
            corresponding_folder = f"FBW10S4_{number}-children"

            # Check if the corresponding folder exists
            if os.path.isdir(os.path.join(directory, corresponding_folder)):
                if max_number is None or number > max_number:
                    max_number = number

    return max_number

def latex_style(enable = True):
    if enable:
        plt.rcParams.update(
                {
                    "font.size": 16,
                    "font.family": "serif",
                    "text.usetex": True,
                    "figure.subplot.top": 0.9,
                    "figure.subplot.right": 0.9,
                    "figure.subplot.left": 0.15,
                    "figure.subplot.bottom": 0.12,
                    "figure.subplot.hspace": 0.4,
                    "savefig.dpi": 200,
                    "savefig.format": "png",
                    "axes.titlesize": 16,
                    "axes.labelsize": 18,
                    "axes.axisbelow": True,
                    "xtick.direction": "in",
                    "ytick.direction": "in",
                    "xtick.major.size": 5,
                    "xtick.minor.size": 2.25,
                    "xtick.major.pad": 7.5,
                    "xtick.minor.pad": 7.5,
                    "ytick.major.pad": 7.5,
                    "ytick.minor.pad": 7.5,
                    "ytick.major.size": 5,
                    "ytick.minor.size": 2.25,
                    "xtick.labelsize": 16,
                    "ytick.labelsize": 16,
                    "legend.fontsize": 16,
                    "legend.framealpha": 1,
                    "figure.titlesize": 16,
                    "lines.linewidth": 2,
                })
    else:
        plt.rcParams.update(plt.rcParamsDefault)
        
def L_to_El(L):
    return (const.hbar/2/const.e)**2/(L)/const.h

def C_to_Ec(C):
    return const.e**2/2/C/const.h

def El_to_L(El):
    return (const.hbar/2/const.e)**2/(El * const.h)

def Ec_to_C(Ec):
    return const.e**2 / (2 * Ec * const.h)

def lorentzian(x, x0, gamma, a):
    return a * gamma**2 / ((x - x0)**2 + gamma**2)

def double_lorentzian(x, x01, gamma1, a1, x02, gamma2, a2, baseline):
    return (lorentzian(x, x01, gamma1, a1) + 
            lorentzian(x, x02, gamma2, a2) + 
            baseline)
    
def triple_lorentzian(x, x01, gamma1, a1, x02, gamma2, a2, x03, gamma3, a3, baseline):
    return (lorentzian(x, x01, gamma1, a1) + 
            lorentzian(x, x02, gamma2, a2) + 
            lorentzian(x, x03, gamma3, a3) + 
            baseline)
    
def preprocess_signal(y_data, x_data):
    """
    Preprocess the signal by filtering out NaNs and interpolating to a uniform time grid.

    Parameters:
    y_data (array-like): The signal data points, potentially containing NaNs.
    time (array-like): The corresponding time points for the signal data.

    Returns:
    tuple: Two arrays, t_uniform and signal_uniform, representing the uniform time grid and the interpolated signal.
    """
    if len(y_data) == 0 or len(x_data) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Filter out NaNs before any processing
    valid_indices = ~np.isnan(y_data)
    y_data = y_data[valid_indices]
    x_data = x_data[valid_indices]

    if len(y_data) < 2:
        raise ValueError("Not enough valid data points after filtering NaNs")

    # Interpolation to convert to a uniform signal
    t_uniform = np.linspace(x_data.min(), x_data.max(), len(x_data))
    interp_func = interp1d(x_data, y_data, kind='linear') #TODO: Compare the interpolation without it.
    signal_uniform = interp_func(t_uniform)

    return t_uniform, signal_uniform


def autocorrelation(signal):
    """
    Calculate the autocorrelation of a signal.

    Parameters:
    signal (array-like): The input signal data points.

    Returns:
    array: The autocorrelation of the input signal.
    """
    n = len(signal)
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[n-1:] / np.arange(n, 0, -1)  # Normalización
    return autocorr

def power_law(f, alpha, c):
    return c**2 * (f**(-alpha))

def fit_power_law(x, y):
    popt, pcov = curve_fit(power_law, x, y)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

def fit_function_with_errors(func, x, y, **kwargs):
    """
    Fits a given function to data and returns the optimized parameters and their errors.

    Parameters:
    func (callable): The model function to be fitted to the data. It should take the independent variable
                     as the first argument and the parameters to be optimized as separate remaining arguments.
    x (array_like): The independent variable where the data is measured.
    y (array_like): The dependent variable, i.e., the data to which the model is fitted.
    **kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit.

    Returns:
    dict: A dictionary with two keys:
          - 'popt': A dictionary of optimized parameter names and their fitted values.
          - 'perr': A dictionary of parameter names and their corresponding standard errors.
    """
    
    # Perform the curve fitting
    popt, pcov = curve_fit(func, xdata=x, ydata=y, **kwargs)
    
    # Get the names of the parameters from the function's signature
    param_names = list(inspect.signature(func).parameters.keys())[1:]
    
    # Create a dictionary with parameter names and their optimized values
    popt_dict = {name: value for name, value in zip(param_names, popt)}
    
    # Calculate the standard errors (square root of the diagonal elements of the covariance matrix)
    perr = np.sqrt(np.diag(pcov))
    perr_dict = {name: error for name, error in zip(param_names, perr)}
    
    return {'popt': popt_dict, 'perr': perr_dict}

