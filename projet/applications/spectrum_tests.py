import numpy as np
import graphinglib as gl

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((4,5), (4,6), 2),
    ],
    10,
    0.1
)

vals, params = spec.evaluate(5)
