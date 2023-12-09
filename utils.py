import inspect
import matplotlib.pyplot as plt
import numpy as np

def filter_args(func, params):
    """ Filtra un diccionario de argumentos para incluir solo los que necesita una función. """
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}

def plot_vs_parameters(x_values_list, y_values_list, parameter_names, ylabels, titles=None, common_title=None, figsize=None, filename=None, single_plot=False, **kwargs):
    plt.close('all')
    
    num_plots = len(x_values_list) if isinstance(x_values_list, list) else x_values_list.ndim
    
    if figsize is None:
        figsize = (6, 6) if num_plots == 1 else (2.5 * num_plots, 6)
    
    fig, ax = plt.subplots(1, num_plots if not single_plot else 1, figsize=figsize, **kwargs)
    
    if num_plots == 1 or single_plot:
        ax = [ax]
        x_values_list = np.array([x_values_list])
        y_values_list = np.array([y_values_list])
        parameter_names = np.array([parameter_names])
        ylabels = np.array([ylabels])
        titles = np.array([titles])
    
    sharey = kwargs.get('sharey', False)
    for i in range(num_plots):
        ax[0 if single_plot else i].plot(x_values_list[i], y_values_list[i])
        
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