import numpy as np
import graphinglib as gl
from eztcolors import Colors as C
from time import time

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import *
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates
from projet.src.fitters.gausspy_fitter import GausspyFitter
from projet.src.fitters.score import *


spec = Spectrum.load("projet/data/spectra/distinct_twin_gaussians/noisy.txt")

# np.random.seed(0)

train_data_array = SpectrumDataArray.generate_from_spectrum(spec, 200)
test_data_array = SpectrumDataArray.generate_from_spectrum(spec, 20)
gf = GausspyFitter(train_data_array, test_data_array)
if __name__ == "__main__":
    print(gf.train_alpha(alpha_initial=1, snr_threshold=3, learning_rate=0.3))

fits = gf.fit(alpha=0.25, snr_threshold=3)

show_fit_plot(test_data_array, fits)

print(mean_r2_score(fits, test_data_array))
