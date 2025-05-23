import numpy as np
import scipy as sp

from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from pathos.pools import ProcessPool


class ScipyFitter:
    """
    This class implements a fitter that uses the scipy.optimize.curve_fit function to fit a model to data. It uses a
    basic algorithm to find peaks in the data and fit a given model to the peaks.
    """

    def __init__(self, data_array: SpectrumDataArray) -> None:
        """
        Initializes a ScipyFitter object.

        Parameters
        ----------
        data_array : SpectrumDataArray
            The data array to fit the model to.
        """
        self.data_array = data_array

    def fit(self, initial_guesses: np.ndarray) -> np.ndarray:
        """
        Fits the data array using scipy.signal.find_peaks and given initial guesses. This method uses multiprocessing
        with pathos and optimized data handling.

        Parameters
        ----------
        initial_guesses : np.ndarray
            The initial guesses for the parameters of the model. The shape is (n,j,k) where n is the number of
            evaluations, j is the number of models and k is the number of parameters per model.

        Returns
        -------
        np.ndarray
            The fitted parameters of the model, given in the same shape as the initial guesses.
        """
        x_values = np.arange(self.data_array.data.shape[1]) + 1

        def worker_fit_single_spectrum(spectrum, guesses):
            # Filter out invalid guesses (rows with np.nan)
            valid_guesses = guesses[~np.isnan(guesses).any(axis=1)]
            if valid_guesses.size == 0:
                # Return NaNs if no valid guesses are available
                return np.full(guesses.size, np.nan)

            # Flatten valid guesses and fit
            try:
                params = sp.optimize.curve_fit(
                    f=self.data_array.spectrum[:valid_guesses.shape[0]],
                    xdata=x_values,
                    ydata=spectrum,
                    p0=valid_guesses.flatten(),
                    maxfev=10000
                )[0]
            except RuntimeError:
                params = np.full(valid_guesses.shape[1], np.nan)

            # Reshape to match the original guesses' shape
            result = np.full(guesses.size, np.nan)
            result[:params.size] = params.flatten()
            return result

        # Pre-pack the arguments to avoid repeated zip overhead
        packed_arguments = [(spectrum, guesses) for spectrum, guesses in zip(self.data_array.data, initial_guesses)]

        with ProcessPool() as pool:
            fit_params = pool.map(lambda args: worker_fit_single_spectrum(*args), packed_arguments)

        return np.array(fit_params).reshape(initial_guesses.shape)
