import numpy as np
import graphinglib as gl
import os

from projet.src.spectrums.spectrum import Spectrum
from projet.src.tools.utilities import show_plot
from projet.src.models.custom_gaussian import CustomGaussian


# Generate a Spectrum with CustomGaussians
spec = Spectrum(
    [
        CustomGaussian((5,10), (20,22), (0.1,0.5)),
        CustomGaussian((6,8), (29,31), (0.1,0.3)),
        CustomGaussian((2,3), (39,41), (0.2,0.4)),
        CustomGaussian((1,2), (55,57), (0.1,0.2)),
        CustomGaussian((1,2), (58,60), (0.1,0.2)),
        CustomGaussian((5,9), (90,94), (0.2,0.5)),
        CustomGaussian((3,5), (75,77), (0.1,0.3)),
    ],
    100,
)

for file, noise_sigma in zip(["smooth", "noisy", "very_noisy"], [0, 0.4, 1]):
    spec.noise_sigma = noise_sigma
    elements = spec.plot
    fig = gl.Figure(); fig.add_elements(*elements); fig.save("test.png")
    input(f"noise_sigma={noise_sigma}, ok ?")
    
    filename = f"projet/data/single_gaussian/{file}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    spec.save(f"{filename}.txt")
    fig.save(f"{filename}.pdf")
