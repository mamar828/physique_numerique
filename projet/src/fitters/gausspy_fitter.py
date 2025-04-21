import numpy as np
import pickle

import projet.src.fitters.gausspy_lib.gp as gp
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.tools.messaging import notify_function_end
from projet.src.tools.array_functions import list_to_array


class GausspyFitter:
    """
    This class implements a fitter that uses the GaussPy library to fit a model to data. It uses the AGD algorithm to
    properly adjust the smoothing parameter alpha that enables the fitter to make the correct initial guesses and fit
    the right number of gaussians.
    For supplementary information, see https://gausspy.readthedocs.io/en/latest/tutorial.html.
    """

    def __init__(self, train_data_array: SpectrumDataArray, test_data_array: SpectrumDataArray) -> None:
        """
        Initializes a GausspyFitter object.

        Parameters
        ----------
        train_data_array : SpectrumDataArray
            The data array to use for training the alpha parameter.
        test_data_array : SpectrumDataArray
            The data array to use for testing the alpha parameter.
        """
        self.train_data_array = train_data_array
        self.test_data_array = test_data_array

    @staticmethod
    def _dump_data(filename: str, data_array: SpectrumDataArray) -> None:
        """
        Dumps a gausspy-readable data array into a pickle file.

        Parameters
        ----------
        filename : str
            The name of the file to dump the data into.
        data_array : SpectrumDataArray
            The data array to dump.
        """
        # The training data first needs to be dumped into a pickle file to be used by the AGD algorithm.
        gp_data = {
            "data_list": [],
            "x_values": [],
            "errors": [np.ones(data_array.spectrum.number_of_channels)] * len(data_array),
            "amplitudes": [],
            "means": [],
            "fwhms": [],
        }

        for spectrum_data, spectrum_params in zip(data_array.data, data_array.params):
            gp_data["data_list"].append(spectrum_data)
            gp_data["x_values"].append(data_array.spectrum.x_values)
            nan_mask = ~np.isnan(spectrum_params).any(axis=1)
            gp_data["amplitudes"].append(spectrum_params[nan_mask, 0])
            gp_data["means"].append(spectrum_params[nan_mask, 1])
            gp_data["fwhms"].append(spectrum_params[nan_mask, 2] * 2*np.sqrt(2*np.log(2)))

        pickle.dump(gp_data, open(filename, "wb"))

    @notify_function_end
    def train_alpha(self, alpha_initial: float=1, snr_threshold: float=3, **kwargs) -> float:
        """
        Trains the smoothing parameter alpha using known underlying gaussian decompositions from the training data
        array.

        Parameters
        ----------
        alpha_initial : float, default=1
            The initial guess for the smoothing parameter alpha.
        snr_threshold : float, default=5
            The signal-to-noise ratio threshold to use for the AGD algorithm. This parameter is used to determine the
            minimum amplitude of the peaks to be considered.
        kwargs
            Additional arguments to pass to the AGD algorithm. These can include parameters such as the maximum number
            of iterations, convergence tolerance, etc.
            See the GaussPy documentation for more details.
        
        Returns
        -------
        float
            The trained smoothing parameter alpha.
        """
        GausspyFitter._dump_data("projet/data/gausspy/training_data.pkl", self.train_data_array)

        # The AGD algorithm is then called to train the smoothing parameter alpha.
        decomposer = gp.GaussianDecomposer()
        decomposer.load_training_data("projet/data/gausspy/training_data.pkl")
        decomposer.set("phase", "one")
        decomposer.set("SNR_thresh", [snr_threshold, snr_threshold])
        decomposer.train(alpha1_initial=alpha_initial, **kwargs, verbose=False)
        return decomposer.p["alpha1"]

    def fit(self, alpha: float, snr_threshold: float=3) -> np.ndarray:
        """
        Fits the testing data array using the trained smoothing parameter alpha.

        Parameters
        ----------
        alpha : float
            The smoothing parameter alpha to use for the fitting.
        snr_threshold : float, default=3
            The signal-to-noise ratio threshold to use for the AGD algorithm. This parameter is used to determine the
            minimum amplitude of the peaks to be considered.

        Returns
        -------
        np.ndarray
            The fitted parameters of the model, with shape (n,j,k) where n is the number of evaluations, j is the number
            of models and k is the number of parameters per model.
        """
        GausspyFitter._dump_data("projet/data/gausspy/testing_data.pkl", self.test_data_array)

        decomposer = gp.GaussianDecomposer()
        decomposer.set("phase", "one")
        decomposer.set("SNR_thresh", [snr_threshold, snr_threshold])
        decomposer.set("alpha1", alpha)

        decomposition = decomposer.batch_decomposition("projet/data/gausspy/testing_data.pkl")

        # Convert the inhomogenous decompositions to rectangular arrays
        amplitudes = list_to_array(decomposition["amplitudes_fit"])
        means = list_to_array(decomposition["means_fit"])
        fwhms = list_to_array(decomposition["fwhms_fit"])

        return np.dstack((amplitudes, means, fwhms / (2*np.sqrt(2*np.log(2)))))
