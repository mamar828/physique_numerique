import numpy as np
import graphinglib as gl
from eztcolors import Colors as C

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates

np.random.seed(0)


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((4,5), (45,55), 2),
        CustomGaussian((7,8), (70,75), 3),
    ],
    100,
    0.0
)
# show_plot(spec.plot)

vals, params = spec.evaluate(5)
# fig = gl.Figure(); fig.add_elements(gl.Curve(np.arange(100) + 1, vals[0])); fig.show()
print(C.LIGHT_GREEN, params, C.END, sep="", end="\n\n")

print(C.LIGHT_PURPLE, find_peaks_gaussian_estimates(vals), C.END, sep="")
