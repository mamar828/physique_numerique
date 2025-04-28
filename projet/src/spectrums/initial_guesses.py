import numpy as np
import scipy as sp

from projet.src.tools.array_functions import list_to_array


def find_peaks_gaussian_estimates(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Finds gaussian initial guesses using a find_peaks algorithm. The parameters should be chosen so the same number of
    peaks are detected in each spectrum.

    Parameters
    ----------
    data : np.ndarray
        A (n,m) numpy array containing n spectra with m channels each.
    kwargs : Any
        Additional arguments to pass to the scipy.signal.find_peaks function.

    Returns
    -------
    np.ndarray
        The initial guesses for the parameters of the Gaussian model. The array has the shape (n,j,3) where n is the
        number of evaluations, j is the number of models and the columns are the estimated amplitude, mean and stddev of
        the Gaussian model.
    """
    peak_means = [sp.signal.find_peaks(spectrum, **kwargs)[0] for spectrum in data]
    peak_amplitudes = [spectrum[peaks] for spectrum, peaks in zip(data, peak_means)]
    peak_means = list_to_array(peak_means)
    peak_amplitudes = list_to_array(peak_amplitudes)
    assert peak_means.size > 0, "No peaks were detected in the data. Please check the parameters passed to find_peaks."

    # Estimate stddevs
    peak_stddevs = []
    for means, amplitude in zip(peak_means.T, peak_amplitudes.T):    # iterate over each detected peak
        half_max_difference = data - amplitude[:,None] / 2
        half_max_intersect_mask = np.abs(np.diff(np.sign(half_max_difference))).astype(bool)
        intersects_x = [np.where(mask)[0] + 1 for mask in half_max_intersect_mask]

        current_stddevs = []
        for intersect, mean in zip(intersects_x, means):
            if np.isnan(mean):
                current_stddevs.append(np.nan)
            else:
                lower_bound_candidates = intersect[intersect < mean]
                lower_bound = lower_bound_candidates.max() if len(lower_bound_candidates) > 0 else 0
                upper_bound_candidates = intersect[intersect > mean]
                upper_bound = upper_bound_candidates.min() if len(upper_bound_candidates) > 0 else 0
                current_stddevs.append((upper_bound - lower_bound) / (2*np.sqrt(2*np.log(2))))

        peak_stddevs.append(current_stddevs)

    peak_stddevs = np.array(peak_stddevs).T
    peak_means += 1     # correct for the 0-based indexing in numpy but 1-based indexing in the data

    return np.dstack((peak_amplitudes, peak_means, peak_stddevs))
