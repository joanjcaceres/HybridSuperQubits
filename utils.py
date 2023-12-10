import inspect
import matplotlib.pyplot as plt
import numpy as np

def filter_args(func, params):
    """
    Filters a dictionary of arguments to include only those that are required by a given function.

    This function examines the signature of the specified function and includes only the arguments
    in the provided dictionary that are necessary for that function. This is useful for dynamically
    determining which arguments to pass to a function when you have more arguments than needed.

    Parameters:
    func (function): The function for which the arguments need to be filtered.
    params (dict): The dictionary of parameters that needs to be filtered.

    Returns:
    dict: A dictionary containing only the arguments that are required by 'func'.
    """

    # Obtain the signature of the function, which includes information about its arguments
    sig = inspect.signature(func)

    # Filter the 'params' dictionary to include only those items whose keys
    # are present in the parameters of the function 'func'
    return {k: v for k, v in params.items() if k in sig.parameters}

def plot_vs_parameters(x_values_list, y_values_list, parameter_names, ylabels, titles=None, common_title=None, figsize=None, filename=None, single_plot=False, **kwargs):
    plt.close('all')
    # if we want to make more than one plot, y_values_list have to be a list.
    num_plots = len(y_values_list) if isinstance(y_values_list, list) else 1
    print(num_plots)
    if figsize is None:
        figsize = (6, 6) if num_plots == 1 else (2.5 * num_plots, 6)
    
    fig, ax = plt.subplots(1, num_plots if not single_plot else 1, figsize=figsize, **kwargs)
    
    if num_plots == 1 or single_plot:
        ax = [ax]
        # x_values_list = np.array([x_values_list])
        y_values_list = np.array([y_values_list])
        parameter_names = np.array([parameter_names])
        ylabels = np.array([ylabels])
        titles = np.array([titles])
    
    sharey = kwargs.get('sharey', False)
    for i in range(num_plots):
        ax[0 if single_plot else i].plot(x_values_list, y_values_list[i])
        
        if not sharey or i == 0:
            ax[0 if single_plot else i].set_ylabel(ylabels[i])
        
        if not single_plot:
            ax[i].set_xlabel(parameter_names[i])
            if titles:
                ax[i].set_title(titles[i])
    
    if single_plot:
        ax[0].set_xlabel(parameter_names[0])
        # ax[0].legend()
        
    if common_title:
        fig.suptitle(common_title)

    fig.tight_layout()
    
    if filename:
        fig.savefig(filename)
    
    plt.show()