import torch
from torch.nn import *

from projet.src.fitters.base_neural_network_fitter import BaseNeuralNetworkFitter


class CNNFitter(BaseNeuralNetworkFitter):
    """
    This class implements a Convolutional Neural Network (CNN) for fitting spectra.
    """

    def __init__(self, number_of_components: int, number_of_channels: int) -> None:
        """
        Constructs a CNNFitter object.

        Parameters
        ----------
        number_of_components : int
            Number of components to fit on the spectrum. Three parameters are fitted for each component: amplitude, mean
            and standard deviation.
        number_of_channels : int
            Number of channels in the spectrum. This number should be a multiple of 4, as the CNN architecture is
            designed to reduce the number of channels by a factor of 4 through max pooling.
        """
        super().__init__()
        self.number_of_components = number_of_components
        self.number_of_channels = number_of_channels

        # Define the CNN architecture
        self.convolution_layers = Sequential(
            Conv1d(1, 16, kernel_size=5, padding=2),
            ReLU(),
            MaxPool1d(2),
            Conv1d(16, 32, kernel_size=5, padding=2),
            ReLU(),
            MaxPool1d(2)
        )
        self.fully_connected_layers = Sequential(
            Flatten(),  # flatten the output from the convolutional layers to a 2D tensor
            Linear(32 * (self.number_of_channels // 4), self.number_of_channels),
            ReLU(), # Softmax(dim=1),
            Linear(self.number_of_channels, 3 * self.number_of_components)  # (amplitude, mean, stddev) per component
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, number_of_channels).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, number_of_components, 3). For each component, the output contains its
            amplitude, mean and standard deviation.
        """
        x = self.convolution_layers(x)
        x = self.fully_connected_layers(x)
        x = x.reshape(x.shape[0], self.number_of_components, 3)
        return x
