import numpy as np
import graphinglib as gl
from eztcolors import Colors as C
from time import time

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import *
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates
from projet.src.fitters.scipy_fitter import ScipyFitter
from projet.src.fitters.score import *


spec = Spectrum.load("projet/data/distinct_gaussians/very_noisy.txt")
# show_plot(spec.plot)

np.random.seed(0)

data_array = SpectrumDataArray.generate_from_spectrum(spec, 10000)
sf = ScipyFitter(data_array)
# estimates = find_peaks_gaussian_estimates(data_array.data, prominence=2)
estimates = find_peaks_gaussian_estimates(data_array.data, prominence=3, height=3, width=5, distance=10)
fits = sf.fit(estimates)

# show_fit_plot(data_array, fits)

# print(mean_squared_error(fits, data_array.params))
print(mean_r2_score(fits, data_array))
