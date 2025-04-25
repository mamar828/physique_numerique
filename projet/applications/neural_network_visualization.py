from torch.utils.data import DataLoader

from projet.src.spectrums.spectrum import Spectrum
from projet.src.data_structures.spectrum_dataset import SpectrumDataset
from projet.src.fitters.cnn_fitter import CNNFitter
from projet.src.fitters.score import *
from projet.src.tools.utilities import show_fit_plot


SPEC_FILE = "distinct_gaussians/very_noisy"
N_SAMPLES = 10000
fitter = CNNFitter.load(f"projet/data/neural_networks/CNNFitter/{SPEC_FILE.replace('/', '_')}_2.pt")

spec = Spectrum.load(f"projet/data/spectra/{SPEC_FILE}.txt")
dataset = SpectrumDataset.generate_from_spectrum(spec, N_SAMPLES)
data_loader = DataLoader(dataset, batch_size=1)

fits = fitter.predict(data_loader)
r2 = mean_r2_score(fits, dataset)
cmse = custom_mean_squared_error(fits, dataset.params)
print(r2, cmse)

# show_fit_plot(dataset, fits, show_true=True)
