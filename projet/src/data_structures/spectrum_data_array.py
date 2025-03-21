import numpy as np
from typing import Self

from projet.src.spectrums.spectrum import Spectrum


class SpectrumDataArray:
    """
    This class implements a data array for spectra that uses numpy arrays to store the intensity data.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initializes a SpectrumDataArray object.

        Parameters
        ----------
        data : np.ndarray
            A size (m, n) numpy array containing m spectra with n channels each.
        """
        self.data = data

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
        data = spectrum.evaluate(n_spectra)
        return cls(data)
