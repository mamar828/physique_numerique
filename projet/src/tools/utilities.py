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
        show_true: bool=True
) -> None:
    """
    Automatically plots the given data array and the fitted parameters.

    Parameters
    ----------
    spectrum_data : SpectrumDataObject
        The data object to plot.
    fits : np.ndarray
        The fitted parameters to plot.
    show_true : bool, default=True
        Whether to show the true parameters or not.
    """
    if isinstance(spectrum_data, SpectrumDataset):
        data = spectrum_data.data.squeeze(1)
    else:
        data = spectrum_data.data
    
    x_space = np.linspace(1, spectrum_data.spectrum.number_of_channels, 1000)
    for spectrum, fit, params in zip(data, fits, spectrum_data.params):
        gl_data = Curve(spectrum_data.spectrum.x_values, spectrum, label="Data")
        gl_fit = Curve(x_space, spectrum_data.spectrum(x_space, fit), line_style=":", label="Fit")
        gl_true = Curve(x_space, spectrum_data.spectrum(x_space, params), line_width=2, label="Real")
        if show_true:
            show_plot(gl_data, gl_true, gl_fit)
        else:
            show_plot(gl_data, gl_fit)
