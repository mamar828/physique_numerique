import numpy as np
from typing import Self

from projet.src.data_structures.spectrum_data_object import SpectrumDataObject
from projet.src.spectrums.spectrum import Spectrum


class SpectrumDataArray(SpectrumDataObject):
    """
    This class implements a data array for spectra that uses numpy arrays to store the intensity data.
    """

    def __init__(self, data: np.ndarray, params: np.ndarray, spectrum: Spectrum) -> None:
        """
        Initializes a SpectrumDataArray object.

        Parameters
        ----------
        data : np.ndarray
            A size (n,m) numpy array containing n spectra with m channels each.
        params : np.ndarray
            A size (n,j,k) numpy array containing the parameters used for creating the data, where n is the number of
            spectra, j is the number of models and k is the number of parameters in each model.
        spectrum : Spectrum
            The Spectrum object used to create the data. This object is used to store the models that generated the data
            and which correspond to the given params.
        """
        self.data = data
        self.params = params
        self.spectrum = spectrum

    @classmethod
    def generate_from_spectrum(cls, spectrum: Spectrum, n_spectra: int) -> Self:
        """
        Create a SpectrumDataArray object from a Spectrum object.

        Parameters
        ----------
        spectrum : Spectrum
            The Spectrum object to create the SpectrumDataArray from.
        n_spectra : int
            The number of spectra to create.

        Returns
        -------
        SpectrumDataArray
            The newly generated SpectrumDataArray object.
        """
        data, params = spectrum.evaluate(n_spectra)
        return cls(data, params, spectrum)
