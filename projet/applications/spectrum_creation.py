import numpy as np
import graphinglib as gl
import os

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((7,10), (20,22), (0.1,0.4)),
        CustomGaussian((2,5), (20,22), (1,5)),
        CustomGaussian((4,7), (70,75), (0.2,1)),
        CustomGaussian((1,3), (70,75), (3,7)),
    ],
    100,
)

for file, noise_sigma in zip(["smooth", "noisy", "very_noisy"], [0, 0.4, 1]):
    spec.noise_sigma = noise_sigma
    elements = spec.plot
    fig = gl.Figure(); fig.add_elements(*elements); fig.save("test.png")
    input(f"noise_sigma={noise_sigma}, ok ?")
    
    filename = f"projet/data/spectra/two_gaussian_components/{file}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    spec.save(f"{filename}.txt")
    fig.save(f"{filename}.pdf")
