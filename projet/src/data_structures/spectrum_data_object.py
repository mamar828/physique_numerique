from numpy import ndarray
from torch import Tensor
from typing import Self, Protocol, runtime_checkable

from projet.src.spectrums.spectrum import Spectrum


@runtime_checkable
class SpectrumDataObject(Protocol):
    """
    This class implements a protocol for spectrum data objects.
    """
    data: ndarray | Tensor
    params: ndarray | Tensor
    spectrum: Spectrum

    def __len__(self) -> int:
        """
        Returns the number of spectra in the SpectrumDataObject object.
        """
        return self.data.shape[0]
    
    def __getitem__(self, index) -> tuple[ndarray | Tensor, ndarray | Tensor]:
        """
        Returns the spectrum and parameters at the given index.
        """
        return self.data[index], self.params[index]

    @classmethod
    def generate_from_spectrum(cls, spectrum: Spectrum, n_spectra: int) -> Self:
        """
        Create a SpectrumDataObject object from a Spectrum object.

        Parameters
        ----------
        spectrum : Spectrum
            The Spectrum object to create the SpectrumDataObject from.
        n_spectra : int
            The number of spectra to create.

        Returns
        -------
        SpectrumDataObject
            The newly generated SpectrumDataObject.
        """
        raise NotImplementedError
