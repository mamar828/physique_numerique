import numpy as np
import pyperclip as pc

from eztcolors import Colors as C
from time import time

import warnings
warnings.simplefilter("ignore")

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import *
from projet.src.models.custom_gaussian import CustomGaussian
from projet.src.data_structures.spectrum_data_array import SpectrumDataArray
from projet.src.spectrums.initial_guesses import find_peaks_gaussian_estimates
from projet.src.fitters.scipy_fitter import ScipyFitter
from projet.src.fitters.score import *


if __name__ == "__main__":
    spectras = [
        #"distinct_gaussians",
        "contaminated_gaussians",
        #"distinct_twin_gaussians",
        #"merged_twin_gaussians",
        "pointy_gaussians",
        "single_gaussian",
        "two_gaussian_components",
    ]
    noise_levels = ["smooth", "noisy", "very_noisy"]
    # 3, 3, 5, 10
    guess_params = {
        "distinct_gaussians": (1.5, 1.5, 2, 10),
        "contaminated_gaussians": (1, 3, 2, 10),
        "distinct_twin_gaussians": (1, 3, 3, 10),
        "merged_twin_gaussians": (0, 2, 3, 2),
        "pointy_gaussians": (1, 1, 0.1, 2),
        "single_gaussian": (0, 0, 5, 80),
        "two_gaussian_components":(1, 1.5, 1, 2), 
    }
    spectras, noise_levels = np.meshgrid(spectras, noise_levels)
    spectras, noise_levels = spectras.flatten(), noise_levels.flatten()

    results = {}
    for nl in noise_levels:
        results[nl] = {}

    for spectra, noise in zip(spectras, noise_levels):
        start = time()
        print(f"{C.YELLOW}{spectra} - {noise}{C.END}:", end=" ")
        # Load the spectrum files
        spec = Spectrum.load(f"projet/data/spectra/{spectra}/{noise}.txt")
        p, h, w, d = guess_params[spectra]

        np.random.seed(2)

        # two components noisy: seed=2
        # contaminated noisy ou very noissy: seed=737, 738
        # single gaussian smooth: seed=0
        #

        num_samples = 1
        data_array = SpectrumDataArray.generate_from_spectrum(spec, num_samples)
        sf = ScipyFitter(data_array)

        estimates = find_peaks_gaussian_estimates(data_array.data, prominence=p, height=h, width=w, distance=d)
        if spectra == "pointy_gaussians":
            estimates[:,:,2] = 0.2
        elif spectra == "two_gaussian_components":
            estimates[:,:,2] = 0.5

        fits = sf.fit(estimates)

        n_line_fitted = np.sum((fits[:,:,0] > 0), axis=1) # .any(axis=2, keepdims=True)
        n_line_true = np.sum((data_array.params[:,:,0] >= 0), axis=1)
        # Percentage of fitted lines
        percentage = 100 * np.count_nonzero(n_line_fitted == n_line_true)/n_line_fitted.size
        print(f"{C.BLUE}good amount of lines fitted: {percentage:.2f} %{C.END},", end=" ")
        fitted_params = fits[n_line_fitted == n_line_true]
        true_params = data_array.params[n_line_fitted == n_line_true]

        if fitted_params.size > 0:
            sorted = np.argsort(fitted_params[:,:,1])
            fitted_params = np.array([
                fitted_params[i][sorted[i]] for i in range(fitted_params.shape[0]) #enumerate(n_line_true) #range(fitted_params.shape[0])
            ])
            fitted_params = fitted_params[:,:n_line_true[0],:]

            sorted = np.argsort(true_params[:,:,1])
            true_params = np.array([
                true_params[i][sorted[i]] for i in range(true_params.shape[0])
            ])
        
        r2 = mean_r2_score(fits, data_array)
        if fitted_params.size > 0:
            mse, mse_list = mean_squared_error(fitted_params, true_params)
        else:
            mse = np.nan
            mse_list = np.array([np.nan])
        print(
            f"{C.GREEN}R^2: {r2}{C.END},",
            f"{C.RED}MSE: {mse}{C.END}," if fitted_params.size > 0 else f"{C.RED}MSE: NaN{C.END},",
            f"{(time() - start)/ num_samples:.4f} s/spectrum"
        )

        results[noise][spectra] = {
            "r2": r2,
            "mse": mse,
            "percentage": percentage,
        }

        show_fit_plot(data_array, fits, show_individual_fits=True, show_total_fit=False, show_true=True)
        show = False
        if fitted_params.size > 0 and show:
            pass
            #print(mse_list[mse_list > 100])
            #show_fit_plot(data_array, fits, index=np.where(n_line_fitted == n_line_true)[0][np.where(mse_list > 100)].reshape(-1), show_individual_fits=True, show_total_fit=False, show_true=True)

    pc.copy(results)
    print("Results copied to clipboard.")
