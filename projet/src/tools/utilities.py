import numpy as np
from graphinglib import Figure, Curve

from projet.src.data_structures.spectrum_data_array import SpectrumDataArray


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

def show_fit_plot(data_array: SpectrumDataArray, fits: np.ndarray) -> None:
    """
    Automatically plots the given data array and the fitted parameters.

    Parameters
    ----------
    data_array : SpectrumDataArray
        The data array to plot.
    fits : np.ndarray
        The fitted parameters to plot.
    """
    for data, fit, param in zip(data_array.data, fits, data_array.params):
        show_plot(
            Curve(data_array.spectrum.x_values, data, label="Data"),
            Curve(data_array.spectrum.x_values, data_array.spectrum(data_array.spectrum.x_values, param),
                  line_width=2, label="Real"),
            Curve(data_array.spectrum.x_values, data_array.spectrum(data_array.spectrum.x_values, fit),
                  line_style=":", label="Fit"),
        )
