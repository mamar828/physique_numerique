import numpy as np
import graphinglib as gl

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((2,8), (17,23), 3),
        CustomGaussian((5,10), (50,60), (5,9)),
        CustomGaussian((0,4), (65,70), (2,3)),
        CustomGaussian((0,3), (85,95), (3,5))
    ],
    100,
    0.1
)
# show_plot(spec.plot)

# Evaluate the Spectrum for 5 spectra and plot the first one
vals = spec.evaluate(5)
fig = gl.Figure()
fig.add_elements(gl.Curve(np.arange(100) + 1, vals[0]))
# fig.show()

# Save the Spectrum to a file and load it back
# spec.save("projet/data/spectrum.txt")
# spec_loaded = Spectrum.load("projet/data/spectrum.txt")
