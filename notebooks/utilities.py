'''
DEPRECATED. I think almost all these functions are already in /src/utilities.
Check if that's the case and eliminate this script.
'''

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import re
# import h5py
# import yaml

# # def load_data(base_file_path):
# #     txt_file_path = base_file_path + '.txt'
# #     hdf5_file_path = base_file_path + '.hdf5'
    
# #     if os.path.exists(txt_file_path):
# #         return load_txt_data(base_file_path)
# #     elif os.path.exists(hdf5_file_path):
# #         return load_hdf5_data(base_file_path)
# #     else:
# #         raise FileNotFoundError(f"Neither {txt_file_path} nor {hdf5_file_path} exist.")


# def load_txt_data(base_name):
#     """
#     Load data from a base name.

#     Parameters:
#     base_name (str): Base name of the files and directory.

#     Returns:
#     dict: Dictionary containing all data columns.
#     """
#     # Define paths to the files and directories based on the base name
#     x_file = f'{base_name}.txt'
#     y_directory = f'{base_name}-children'

#     # Initialize the dictionary
#     data_dict = {}

#     # Load X-axis data
#     with open(x_file, 'r') as file:
#         lines = file.readlines()

#     # Extract parameters from the header
#     parameters = read_parameters(lines)
#     if parameters:
#         data_dict['parameters'] = parameters

#     # Find the end of the header and read column headers
#     header_end_idx = 0
#     for i, line in enumerate(lines):
#         if line.strip() == '#end of header':
#             header_end_idx = i
#             break

#     # Read the column headers
#     column_headers = lines[header_end_idx + 1].strip().split()

#     # Read the data after the header
#     data = [line.strip().split() for line in lines[header_end_idx + 2:] if line.strip()]
#     data = np.array(data, dtype=float)

#     num_rows = data.shape[0]

#     # Populate the data dictionary with columns from x_file
#     for i, header in enumerate(column_headers):
#         data_dict[header] = data[:, i]

#     # Check if the directory exists
#     if os.path.isdir(y_directory):
#         filenames = sorted([f for f in os.listdir(y_directory) if f.endswith('.txt')])

#         mask = []

#         for idx, filename in enumerate(filenames):
#             if idx >= num_rows:
#                 break
#             y_file = os.path.join(y_directory, filename)
#             with open(y_file, 'r') as file:
#                 lines = file.readlines()

#             # Find the end of the header
#             header_end_idx = 0
#             for i, line in enumerate(lines):
#                 if line.strip() == '#end of header':
#                     header_end_idx = i
#                     break

#             # Read the column headers
#             file_column_headers = lines[header_end_idx + 1].strip().split()

#             # Read the data after the header
#             data = [line.strip().split() for line in lines[header_end_idx + 2:] if line.strip()]
#             data = np.array(data, dtype=float)
#             if idx == 0:
#                 correct_shape = data.shape
#             if not data.shape == correct_shape:
#                 mask.append(idx)

#             # Populate the data dictionary with columns from y_directory
#             for i, header in enumerate(file_column_headers):
#                 if header not in data_dict:
#                     data_dict[header] = []
#                 if data.shape == correct_shape:
#                     data_dict[header].append(data[:, i])

#         for key in data_dict:
#             if isinstance(data_dict[key], list):
#                 data_dict[key] = np.array(data_dict[key])

#     for i, header in enumerate(column_headers):
#         if isinstance(data_dict[header], np.ndarray):
#             data_dict[header] = np.delete(data_dict[header], mask, axis=0)

#     return data_dict

# def load_hdf5_data(file_path):
#     hdf5_file_path = f"{file_path}.hdf"
#     with h5py.File(hdf5_file_path, 'r') as f:
#         # Carga los parámetros y meta atributos
#         parameters = yaml.safe_load(filter_lines(f.attrs['parameters']))
#         meta = yaml.safe_load(filter_lines(f.attrs['meta']))
        
#         # Extrae el fieldMap del meta principal
#         main_field_map = meta.get('fieldMap', {})
        
#         # Inicializa el diccionario de datos con los parámetros
#         data_dict = {'parameters': parameters}
        
#         # Si existe un table en el grupo principal, procesarlo
#         if 'table' in f:
#             main_table = f['table']
#             for field, index in main_field_map.items():
#                 data_dict[field] = []
#                 for row_idx in range(main_table.shape[0]):
#                     data_dict[field].append(main_table[row_idx, index])
        
#         # Ordena los nombres de los subgrupos numéricamente
#         subgroups_sorted = sorted(f['children'].keys(), key=lambda x: int(x))
#         for idx, group in enumerate(subgroups_sorted):
#             table = f['children'][group]['table']
#             if idx == 0:
#                 # Obtiene el meta atributo del primer subgrupo para el fieldMap del subgrupo
#                 meta_child = yaml.safe_load(filter_lines(f['children'][group].attrs['meta']))
#                 child_field_map = meta_child['fieldMap']
                
#                 # Inicializa las listas para cada campo del fieldMap del subgrupo
#                 for field in child_field_map:
#                     if field not in data_dict:
#                         data_dict[field] = []
                    
#             # Extrae los datos del dataset y los agrega a las listas correspondientes
#             for field, index in child_field_map.items():
#                 if len(data_dict[field]) == 0:
#                     # Inicializa la lista de listas si es la primera vez
#                     data_dict[field] = [[] for _ in range(table.shape[0])]
#                 # Agrega los datos a la estructura correcta
#                 for row_idx in range(table.shape[0]):
#                     data_dict[field][row_idx].append(table[row_idx, index])
        
#         # Convertir las listas a arrays numpy y asegurarse de que la forma sea correcta
#         for field in data_dict:
#             if field != 'parameters':
#                 data_dict[field] = np.array(data_dict[field])
            
#     return data_dict

# def filter_lines(yaml_str):
#     filtered_lines = []
#     for line in yaml_str.split('\n'):
#         if '!!python/' not in line:
#             filtered_lines.append(line)
#     return '\n'.join(filtered_lines)

# def plot_phase_map(data_dict, x_key, y_key, z_key, title='Phase Map', correct_horizontal=False, correct_vertical=False, fig=None, ax=None, **kwargs):
#     """
#     Plot a 2D phase map using the provided X-axis values, Y-axis values, and Z-axis matrix.

#     Parameters:
#     data_dict (tuples): Tuples containing (x_values, y_values, Z_values).
#     x_key (str): Key for the x_values to plot.
#     y_key (str): Key for the y_values to plot.
#     z_key (str): Key for the z_values to plot.
#     title (str): Title of the plot.
#     correct_horizontal (bool): Whether to apply horizontal correction.
#     correct_vertical (bool): Whether to apply vertical correction.
#     fig (matplotlib.figure.Figure, optional): Figure object to use for plotting.
#     ax (matplotlib.axes.Axes, optional): Axes object to use for plotting.
#     kwargs: Additional keyword arguments to pass to pcolormesh.
#     """
#     if fig is None or ax is None:
#         figsize= kwargs.pop('figsize', (8, 6))
#         fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis if not provided
    
#     x_data = data_dict[x_key]
#     y_data = data_dict[y_key]
#     z_data = data_dict[z_key]

#     # Apply corrections if specified
#     if x_data.ndim == 1:
#         Y = y_data
#         X = np.tile(x_data, (Y.shape[1], 1)).T
#         if correct_horizontal:
#             z_data = average_and_subtract_horizontal(z_data)
#         if correct_vertical:
#             z_data = average_and_subtract_vertical(z_data)
#     elif y_data.ndim == 1: # In case it's inverted the axis.
#         X = x_data
#         Y = np.tile(y_data, (X.shape[1], 1)).T
#         if correct_vertical:
#             z_data = average_and_subtract_horizontal(z_data)
#         if correct_horizontal:
#             z_data = average_and_subtract_vertical(z_data)

#     # Plot using pcolormesh
#     vmin = kwargs.pop('vmin', None)
#     vmax = kwargs.pop('vmax', None)
#     mesh = ax.pcolormesh(X, Y, z_data, shading='auto', vmin=vmin, vmax=vmax,**kwargs)
    
#     ax.set_title(title)
#     ax.set_xlabel(x_key)
#     ax.set_ylabel(y_key)
#     fig.tight_layout()
    
#     return fig, ax, mesh

# def average_and_subtract_vertical(phase_matrix):
#     """
#     Calculate the vertical average and subtract it from each row in the phase matrix.

#     Parameters:
#     phase_matrix (np.ndarray): Matrix of phase values.

#     Returns:
#     np.ndarray: Phase matrix with vertical averages subtracted.
#     """
#     vertical_mean = np.nanmean(phase_matrix, axis=1)  # Calculate mean of each row
#     phase_matrix_corrected = phase_matrix - vertical_mean[:, np.newaxis]  # Subtract the mean from each row
#     return phase_matrix_corrected

# def average_and_subtract_horizontal(phase_matrix):
#     """
#     Calculate the horizontal average and subtract it from each column in the phase matrix.

#     Parameters:
#     phase_matrix (np.ndarray): Matrix of phase values.

#     Returns:
#     np.ndarray: Phase matrix with horizontal averages subtracted.
#     """
#     horizontal_mean = np.nanmean(phase_matrix, axis=0)  # Calculate mean of each column
#     phase_matrix_corrected = phase_matrix - horizontal_mean  # Subtract the mean from each column
#     return phase_matrix_corrected

# def plot_1d_line(datasets, x_key, y_key, z_key, fixed_x=None, fixed_y=None, tolerance=0.01, apply_vertical_correction=False, apply_horizontal_correction=False, 
#                  fig=None, ax=None, **kwargs):
#     """
#     Plot a 1D line at a fixed Y value on an existing phase map.

#     Parameters:
#     datasets (list of dict): List of dictionaries containing all data columns.
#     fixed_y (float): Fixed Y value to plot a 1D line.
#     x_key (str): Key for the x_values to plot.
#     y_key (str): Key for the y_values to plot.
#     z_key (str): Key for the z_values to plot.
#     tolerance (float): Tolerance within which the nearest Y value must be to fixed_y.
#     apply_vertical_correction (bool): Whether to apply vertical correction to the z_values.
#     apply_horizontal_correction (bool): Whether to apply horizontal correction to the z_values.
#     fig (matplotlib.figure.Figure, optional): Figure object to use for plotting.
#     ax (matplotlib.axes.Axes, optional): Axes object to use for plotting.
#     kwargs: Additional keyword arguments to pass to the plot function.
#     """
#     if fig is None or ax is None:
#         figsize =kwargs.pop('figsize', (8,6))
#         fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis if not provided
    
#     if fixed_x is not None and fixed_y is not None:
#         raise ValueError("Both fixed_x and fixed_y shouldn't be different to None.")
    
#     x_data = datasets[x_key]
#     y_data = datasets[y_key]
#     z_data = datasets[z_key]

#     if x_data.ndim == 1:
#         Y = y_data
#         X = np.tile(x_data, (Y.shape[1], 1)).T
#         if apply_horizontal_correction:
#             z_data = average_and_subtract_horizontal(z_data)
#         if apply_vertical_correction:
#             z_data = average_and_subtract_vertical(z_data)
#     elif y_data.ndim == 1: # In case it's inverted the axis.
#         X = x_data
#         Y = np.tile(y_data, (X.shape[1], 1)).T
#         if apply_vertical_correction:
#             z_data = average_and_subtract_horizontal(z_data)
#         if apply_horizontal_correction:
#             z_data = average_and_subtract_vertical(z_data)

#     #This only works when x_data is 1D array.

#     if fixed_y is not None:
#         y_index = (np.abs(Y[0,:] - fixed_y)).argmin()
#         z_1d = z_data[y_index,:]

#         y_nearest = Y[y_index,0]
#         if np.abs(y_nearest - fixed_y) > tolerance:
#             print(f"Skipping dataset with y_nearest = {y_nearest:.2f} (outside tolerance of {tolerance})")


#         ax.plot(x_data[0,:], z_1d, label=f'Y = {y_nearest:.2f}', **kwargs)
#         ax.set_xlabel(x_key)
#         ax.set_ylabel(z_key)
#     elif fixed_x is not None:
#         x_index = (np.abs(X[:,0] - fixed_x)).argmin()
#         z_1d = z_data[x_index, :]

#         x_nearest = X[x_index,0]
#         if np.abs(x_nearest - fixed_x) > tolerance:
#             print(f"Skipping dataset with x_nearest = {x_nearest:.2f} (outside tolerance of {tolerance})")

#         ax.plot(y_data[0,:], z_1d, label=f'X = {x_nearest:.2f}', **kwargs)
#         ax.set_xlabel(y_key)
#         ax.set_ylabel(z_key)

#     ax.legend()
#     fig.tight_layout()
#     return fig, ax

# # def read_parameters(lines):
# #     in_parameters_section = False
# #     parameters_lines = []

# #     for line in lines:
# #         if line.strip().startswith('parameters:'):
# #             in_parameters_section = True
# #             # Remove 'parameters: {' and any surrounding spaces
# #             parameters_lines.append(line.strip().replace('parameters: {', '').strip())
# #         elif in_parameters_section:
# #             # Remove the closing brace and any surrounding spaces
# #             if '}' in line:
# #                 parameters_lines.append(line.strip().replace('}', '').strip())
# #                 break
# #             else:
# #                 parameters_lines.append(line.strip())

# #     if parameters_lines:
# #         # Join all the lines of parameters into a single string
# #         parameters_str = ' '.join(parameters_lines)
        
# #         # Convert the string into a dictionary
# #         parameters = {}
# #         items = parameters_str.split(', ')
# #         for item in items:
# #             key_value = item.split(': ')
# #             if len(key_value) == 2:
# #                 key, value = key_value
# #                 # Convert values to float where possible
# #                 try:
# #                     value = float(value)
# #                 except ValueError:
# #                     pass
# #                 parameters[key] = value

# #         return parameters
# #     else:
# #         return None

# def last_measurument(directory):
#     max_number = None
#     # Regex to match the file pattern
#     file_pattern = re.compile(r'FBW10S4_(\d+)\.txt')

#     for filename in os.listdir(directory):
#         # Check if the filename matches the pattern
#         match = file_pattern.match(filename)
#         if match:
#             number = int(match.group(1))
#             corresponding_folder = f"FBW10S4_{number}-children"

#             # Check if the corresponding folder exists
#             if os.path.isdir(os.path.join(directory, corresponding_folder)):
#                 if max_number is None or number > max_number:
#                     max_number = number

#     return max_number


