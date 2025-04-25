import torch
import numpy as np
from typing import Self, Iterable

from projet.src.spectrums.spectrum import Spectrum
from projet.src.data_structures.spectrum_data_object import SpectrumDataObject


class SpectrumDataset(SpectrumDataObject):
    """
    This class implements a dataset for spectra that uses PyTorch tensors to store the intensity data.
    """

    def __init__(self, data: torch.Tensor, params: torch.Tensor, spectrum: Spectrum) -> None:
        """
        Initializes a SpectrumDataset object.

        Parameters
        ----------
        data : torch.Tensor
            A size (n,m) tensor containing n spectra with m channels each.
        params : torch.Tensor
            A size (n,j,k) tensor containing the parameters used for creating the data, where n is the number of
            spectra, j is the number of models and k is the number of parameters in each model.
        spectrum : Spectrum
            The Spectrum object used to create the data. This object is used to store the models that generated the data
            and which correspond to the given params.
        """
        self.data = data
        self.params = params
        self.spectrum = spectrum

    @classmethod
    def generate_from_spectrum(cls, spectrum: Spectrum, n_spectra: int) -> Self:
        """
        Create a SpectrumDataset object from a Spectrum object.

        Parameters
        ----------
        spectrum : Spectrum
            The Spectrum object to create the SpectrumDataset from.
        n_spectra : int
            The number of spectra to create.

        Returns
        -------
        SpectrumDataset
            The newly generated SpectrumDataset object.
        """
        data, params = spectrum.evaluate(n_spectra)

        tensor_data = torch.tensor(data, dtype=torch.float32)
        tensor_params = torch.tensor(params, dtype=torch.float32)
        if data.ndim == 2:
            tensor_data = tensor_data.unsqueeze(1)
        
        return cls(tensor_data, tensor_params, spectrum)

    def random_split(self, lengths: Iterable[float | int]) -> list[Self]:
        """
        Splits the SpectrumDataset into random Datasets of the given lengths. This method is equivalent to the
        torch.utils.data.random_split method, but returns SpectrumDataset instances instead of Subset instances. It also
        allows to specify the size of each dataset directly.

        Parameters
        ----------
        lengths : Iterable[float | int]
            If the values are floats, their sum should amount to 1.0 and they correspond to the proportion of each
            dataset relative to the total length of the input dataset.
            If the values are ints, they correspond to the exact number of samples to take from each dataset.
        
        Returns
        -------
        list[Self]
            SpectrumDataset instances of the given lengths.
        """
        if np.isclose(sum(lengths), 1.0):
            split_quantities = (np.array(lengths) * len(self)).astype(int)   # get the desired length of each dataset
        else:
            split_quantities = lengths
            
        split_indices = [0] + list(np.cumsum(split_quantities))
        slices = [slice(split_indices[i], split_indices[i+1]) for i in range(len(lengths))]
        new_datasets = [self.__class__(*self[s], self.spectrum) for s in slices]
        return new_datasets
