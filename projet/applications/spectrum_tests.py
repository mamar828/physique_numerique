import numpy as np
import graphinglib as gl
from eztcolors import Colors as C
from time import time

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates
from projet.src.fitters.scipy_fitter import ScipyFitter
from projet.src.fitters.score import mean_squared_error


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((4,5), (45,55), 2),
        CustomGaussian((7,8), (70,75), 3),
    ],
    100,
    0.17
)
# show_plot(spec.plot)

# x_values = np.arange(100) + 1
# vals, params = spec.evaluate(1000)
# initial_guesses = find_peaks_gaussian_estimates(vals, prominence=1)
# for vals_i, guesses in zip(vals, initial_guesses):
#     fig = gl.Figure()
#     fig.add_elements(
#         gl.Curve(x_values, vals_i, color="black", label="Real data"),
#         gl.Scatter(guesses[:,1], guesses[:,0], face_color="green", label="Initial guesses"),
#         gl.Curve(x_values, spec(x_values, guesses), label="Initial guess model")
#     )
#     fig.show()

data_array = SpectrumDataArray.generate_from_spectrum(spec, 10000)
sf = ScipyFitter(data_array)
estimates = find_peaks_gaussian_estimates(data_array.data, prominence=2, height=3, width=3, distance=10)
fits = sf.fit(estimates)

print(mean_squared_error(fits, data_array.params))
