from torch.utils.data import DataLoader

from projet.src.spectrums.spectrum import Spectrum
from projet.src.data_structures.spectrum_dataset import SpectrumDataset
from projet.src.fitters.cnn_fitter import CNNFitter
from projet.src.fitters.score import mean_r2_score
from projet.src.tools.utilities import show_fit_plot


# -----TRAINING PARAMETERS-----
N_EPOCHS = 10
BATCH_SIZE = 500
NUM_SAMPLES = 100000
train_test_split = [0.8, 0.2]
# -----------------------------

spec = Spectrum.load("projet/data/spectra/distinct_gaussians/very_noisy.txt")
dataset = SpectrumDataset.generate_from_spectrum(spec, NUM_SAMPLES)
train_set, test_set = dataset.random_split(train_test_split)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=1)

# fitter = CNNFitter(number_of_components=len(spec.models), number_of_channels=spec.number_of_channels)
# fitter.training_loop(
#     train_loader=train_loader,
#     n_epochs=N_EPOCHS,
# )
# fitter.save("projet/data/neural_networks/cnn/distinct_gaussians_very_noisy.pt")

fitter = CNNFitter.load("projet/data/neural_networks/cnn/distinct_gaussians_very_noisy.pt")
fits = fitter.predict(test_loader)
print(f"Mean R^2 score: {mean_r2_score(fits, test_set)}")

# show_fit_plot(test_set, fits, show_true=False)
