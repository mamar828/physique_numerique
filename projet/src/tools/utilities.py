import numpy as np
from graphinglib import Figure, Curve

from projet.src.data_structures.spectrum_data_object import SpectrumDataObject
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.data_structures.spectrum_dataset import SpectrumDataset


def format_time(total_seconds: float, precision: int=2) -> str:
    """
    Format a given time in seconds into a human-readable string. An example of the output format is "13h05m06.23s".

    Parameters
    ----------
    total_seconds : float
        The total time in seconds to be formatted.
    precision : int, default=2
        The number of decimal places for the seconds.

    Returns
    -------
    str
        A human-readable string representing the formatted time.
    """
    end_str = ""
    # Explicitly convert the numbers to int as python's "integer division" does not always output integer results...
    hours, minutes, seconds = int(total_seconds//3600), int((total_seconds%3600)//60), (total_seconds%3600)%60
    if hours:
        end_str += f"{hours}h"
    if minutes:
        if hours:
            # Force the minutes format to have two digits
            end_str += f"{minutes:02d}m"
        else:
            end_str += f"{minutes}m"
        # Force the seconds format to have two digits
        end_str += f"{seconds:05.{precision}f}s"
    else:
        end_str += f"{seconds:.{precision}f}s"
    return end_str

def show_plot(*plottables) -> None:
    """
    Automatically plots the given plottables and shows the figure.

    Parameters
    ----------
    plottables
        The list of plottables to plot.
    """
    fig = Figure()
    fig.add_elements(*plottables)
    fig.show()

def show_fit_plot(
        spectrum_data: SpectrumDataObject, 
        fits: np.ndarray, 
        show_total_fit: bool=True,
        show_true: bool=True,
        show_individual_fits: bool=False,
        index: np.ndarray=None
) -> None:
    """
    Automatically plots the given data array and the fitted parameters.

    Parameters
    ----------
    spectrum_data : SpectrumDataObject
        The data object to plot.
    fits : np.ndarray
        The fitted parameters to plot.
    show_total_fit : bool, default=True
        Whether to show the total fit or not.
    show_true : bool, default=True
        Whether to show the true parameters or not.
    show_individual_fits : bool, default=False
        Whether to show the individual fits or not. If False, only the global fit is shown (sum of each individual fit).
    """
    
    if index is None:
        index = np.arange(spectrum_data.data.shape[0])
    if isinstance(spectrum_data, SpectrumDataset):
        data = spectrum_data.data.squeeze(1)
    else:
        data = spectrum_data.data[index]
    
    #print(fits[index])
    #print(spectrum_data.params[index])
    
    x_space = np.linspace(1, spectrum_data.spectrum.number_of_channels, 1000)
    for spectrum, fit, params in zip(data, fits[index], spectrum_data.params[index]):
        plottables = [Curve(spectrum_data.spectrum.x_values, spectrum, label="Data")]
        if show_true:
            plottables.append(Curve(x_space, spectrum_data.spectrum(x_space, params), line_width=2, label="Real"))
        if show_individual_fits:
            plottables.extend([Curve(x_space, model(x_space, *fit_i)) 
                               for model, fit_i in zip(spectrum_data.spectrum.models, fit)])
        if show_total_fit:
            # Draw the global fit last to place it on top of the other plottables
            plottables.append(Curve(x_space, spectrum_data.spectrum(x_space, fit), line_style=":", label="Fit"))
            
        show_plot(*plottables)
