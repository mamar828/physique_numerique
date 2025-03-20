from typing import Self
import numpy as np
from astropy.modeling.core import Fittable1DModel
from astropy.modeling import models

from projet.src.spectrums.spectrum import Spectrum


class ArtificialSpectrum(Spectrum):
    """
    This class implements artificial spectra.
    """

    def __init__(self, data: np.ndarray, models: list[Fittable1DModel], noise_sigma: float):
        """
        Initialize an ArtificialSpectrum object.

        Parameters
        ----------
        data : np.ndarray
            The values of the spectrum.
        models : list[Fittable1DModel]
            The list of models that were used to create the spectrum.
        noise_sigma : float
            The standard deviation of the Gaussian noise added to the spectrum.
        """
        super().__init__(data)
        self.models = models
        self.noise_sigma = noise_sigma

    @classmethod
    def generate_from_models(cls, models: list[Fittable1DModel], number_of_channels: int, noise_sigma: float=0) -> Self:
        """
        Create an ArtificialSpectrum object from a list of Gaussian1D models. Note that the first channel starts at 1.

        Parameters
        ----------
        models : list[Fittable1DModel]
            The list of models that are used.
        number_of_channels : int
            The number of channels to create the spectrum
        noise_sigma : float, default=0
            The standard deviation of the Gaussian noise to add to the spectrum.

        Returns
        -------
        ArtificialSpectrum
            The newly generated ArtificialSpectrum object.
        """
        x = np.arange(number_of_channels) + 1       # first channel starts at 1
        data = sum([model(x) for model in models])
        data += np.random.normal(0, noise_sigma, number_of_channels)
        return cls(data, models, noise_sigma)


ArtificialSpectrum.generate_from_models(
    [
        models.Gaussian1D(mean=15, amplitude=10, stddev=3),
        models.Gaussian1D(mean=70, amplitude=15, stddev=4),
        models.Gaussian1D(mean=22, amplitude=5, stddev=3)
    ], 
    number_of_channels=100,
    noise_sigma=0.3
).auto_plot()
