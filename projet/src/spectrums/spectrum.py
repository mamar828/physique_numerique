import numpy as np
from numpy import random
from graphinglib import Curve, Scatter, Line, get_colors
from itertools import cycle
from typing import Self
from os.path import isfile
from eztcolors import Colors as C

from projet.src.models.custom_model import CustomModel


class Spectrum:
    """
    This class implements a spectrum created from models and may be given to a SpectrumDataArray or SpectrumDataset to
    generate corresponding data.
    """

    def __init__(
            self,
            models: list[CustomModel],
            number_of_channels: int,
            noise_sigma: float=0
    ):
        """
        Initializes a Spectrum object.

        Parameters
        ----------
        models : list[CustomModel]
            The list of models to create the spectrum.
        number_of_channels : int
            The number of channels to create the spectrum. Note that the channels are indexed starting from 1.
        noise_sigma : float, default=0
            The standard deviation of the Gaussian noise added to the spectrum.
        """
        self.models = models
        self.number_of_channels = number_of_channels
        self.noise_sigma = noise_sigma

    def __str__(self) -> str:
        """
        Gives a string representation of the Spectrum object.
        """
        return f"Spectrum with {len(self.models)} models, {self.number_of_channels} channels and a noise with a " \
               f"{self.noise_sigma} sigma."
    
    @staticmethod
    def load(filename: str) -> Self:
        """
        Loads a Spectrum object from a file.

        Parameters
        ----------
        filename : str
            The filename to load the Spectrum object from.

        Returns
        -------
        Spectrum
            The loaded Spectrum object.
        """
        with open(filename, "r") as f:
            code = "".join(f.readlines()[1:])
            
        return eval(code)

    @property
    def plot(self) -> list[Curve | Scatter | Line]:
        """
        Gives the plot of the spectrum.

        Returns
        -------
        list[Curve | Scatter | Line]
            The plot of the spectrum.
        """
        x = np.arange(self.number_of_channels) + 1       # first channel starts at 1
        data = np.zeros(self.number_of_channels)
        plottables = []
        for model, color in zip(self.models, cycle(get_colors())):
            data += model(x)
            plottables.extend(model.get_plot(self.number_of_channels, color))

        # Applying Gaussian noise to the data
        data += random.normal(0, self.noise_sigma, self.number_of_channels)
        plottables.append(
            Line(
                [1, -self.noise_sigma*3], 
                [1, self.noise_sigma*3],
                color="gray",
                width=2,
                capped_line=True,
                cap_width=0.3
            )
        )

        return [Curve(x, data, color="black"), *plottables]

    def evaluate(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates n times the spectrum at the given x. This uses the evaluate method of each model in the spectrum,
        which randomly samples the parameters from the model's parameter distribution.

        Parameters
        ----------
        n : int
            The number of times to evaluate the model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The evaluated models at x and the parameters used to evaluate the models. The first array is the sum of all
            the models' data and the second array is the concatenated parameters of each model.
        """
        x = np.tile(np.arange(self.number_of_channels) + 1, (n, 1))     # first channel starts at 1
        params = np.empty((n, 0))

        # Create noisy base data
        data = np.random.normal(0, self.noise_sigma, (n, self.number_of_channels))

        # Add each model's contribution
        for model in self.models:
            model_data, model_params = model.evaluate(x)
            data += model_data
            params = np.hstack((params, model_params))
        
        return data, params

    def save(self, filename: str):
        """
        Saves the spectrum to a file.

        Parameters
        ----------
        filename : str
            The filename to save the spectrum to.
        """
        if isfile(filename):
            input(f"{C.LIGHT_RED}File {filename} already exists. Press Enter to overwrite or Ctrl+C to cancel.{C.END}")
        with open(filename, "w") as f:
            print(self, file=f)
            f.write(f"{self.__class__.__name__}(\n\tmodels=[\n\t\t")
            f.write(",\n\t\t".join([str(model) for model in self.models]))
            f.write(f"\n\t],\n\tnumber_of_channels={self.number_of_channels},")
            f.write(f"\n\tnoise_sigma={self.noise_sigma}\n)")
