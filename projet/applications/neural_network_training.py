from torch.utils.data import DataLoader
from time import time
from datetime import datetime
import os

from projet.src.spectrums.spectrum import Spectrum
from projet.src.data_structures.spectrum_dataset import SpectrumDataset
from projet.src.fitters.cnn_fitter import CNNFitter
from projet.src.fitters.score import *
from projet.src.tools.utilities import format_time


# -----TRAINING PARAMETERS-----
SPEC_FILE = "distinct_gaussians/very_noisy"
N_EPOCHS = 2
BATCH_SIZE = 500
N_SAMPLES = 1000
train_test_split = [0.8, 0.2]
# -----------------------------

spec = Spectrum.load(f"projet/data/spectra/{SPEC_FILE}.txt")
dataset = SpectrumDataset.generate_from_spectrum(spec, N_SAMPLES)
train_set, test_set = dataset.random_split(train_test_split)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=1)

fitter = CNNFitter(number_of_components=len(spec.models), number_of_channels=spec.number_of_channels)
start_time = time()
fitter.training_loop(
    train_loader=train_loader,
    n_epochs=N_EPOCHS,
)
stop_time = time()

fitter_class_name = fitter.__class__.__name__
SAVE_FILE = f"{fitter_class_name}/{SPEC_FILE.replace('/', '_')}"
# os.makedirs(f"projet/data/neural_networks/{fitter_class_name}", exist_ok=True)

fitter.save(f"projet/data/neural_networks/{SAVE_FILE}.pt")

fits = fitter.predict(test_loader)
r2 = mean_r2_score(fits, test_set)
mse = mean_squared_error(fits, test_set.params)

with open(f"projet/data/neural_networks/{fitter_class_name}/info.csv", "a") as f:
    if f.tell() == 0:   # if the file is empty, write the header
        f.write(f"file,n_epochs,batch_size,n_train_samples,n_test_samples,R^2,MSE,training_time,date\n")

    f.write(f"{SPEC_FILE},{N_EPOCHS},{BATCH_SIZE},{len(train_set)},{len(test_set)},{r2:.6f},{mse:.6f},"
            f"{format_time(stop_time-start_time)},{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n")
print(f"R^2: {r2:.6f}, MSE: {mse:.6f}")
