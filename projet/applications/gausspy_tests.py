import numpy as np
import graphinglib as gl
from eztcolors import Colors as C
from time import time

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import *
from projet.src.tools.messaging import *
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates
from projet.src.fitters.gausspy_fitter import GausspyFitter
from projet.src.fitters.score import *


if __name__ == "__main__":
    spec = Spectrum.load("projet/data/spectra/contaminated_gaussians/noisy.txt")

    # np.random.seed(0)

    train_data_array = SpectrumDataArray.generate_from_spectrum(spec, 1000)
    test_data_array = SpectrumDataArray.generate_from_spectrum(spec, 1000)
    gf = GausspyFitter(train_data_array, test_data_array)

    # alpha = gf.train_alpha(alpha_initial=1, snr_threshold=3, learning_rate=0.1)
    # print(alpha)
    # alpha = 0.2579786783847283
    alpha = 0.16352

    fits = gf.fit(alpha=alpha, snr_threshold=3)

    show_fit_plot(test_data_array, fits)

    score = mean_r2_score(fits, test_data_array)
    # print(score)
    # telegram_send_message(f"Mean R$^2$ score: {score:.3f} with alpha : {alpha:.5f}")
