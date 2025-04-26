from torch.utils.data import DataLoader

from projet.src.spectrums.spectrum import Spectrum
from projet.src.data_structures.spectrum_dataset import SpectrumDataset
from projet.src.fitters.cnn_fitter import CNNFitter
from projet.src.fitters.res_net_fitter import ResNetFitter
from projet.src.fitters.score import *
from projet.src.tools.utilities import show_fit_plot


SPEC_FILE = "two_gaussian_components/very_noisy"
N_SAMPLES = 200
fitter = ResNetFitter.load(f"projet/data/neural_networks/ResNetFitter/{SPEC_FILE.replace('/', '_')}.pt")

spec = Spectrum.load(f"projet/data/spectra/{SPEC_FILE}.txt")
dataset = SpectrumDataset.generate_from_spectrum(spec, N_SAMPLES)
data_loader = DataLoader(dataset, batch_size=1)


fits = fitter.predict(data_loader)

show_fit_plot(dataset, fits, show_true=True, show_individual_fits=True, show_total_fit=True)
