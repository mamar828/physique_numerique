import torch
from torch.nn import *

from projet.src.fitters.base_neural_network_fitter import BaseNeuralNetworkFitter


class ResidualBlock(Module):
    """
    This class implements a residual block with two convolutional layers.
    """
    def __init__(self, in_out_channels: int, kernel_size: int = 5) -> None:
        """
        Constructs a ResidualBlock object.

        Parameters
        ----------
        in_out_channels : int
            Number of input and output channels for the convolutional layers.
        kernel_size : int, default=5
            Size of the kernel for the convolutional layers.
        """
        super().__init__()
        padding = kernel_size // 2
        self.block = Sequential(
            Conv1d(in_out_channels, in_out_channels, kernel_size=kernel_size, padding=padding),
            BatchNorm1d(in_out_channels),
            ReLU(),
            Conv1d(in_out_channels, in_out_channels, kernel_size=kernel_size, padding=padding),
            BatchNorm1d(in_out_channels),
        )
        self.activation = ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        """
        return self.activation(x + self.block(x))  # Residual connection


class ResNetFitter(BaseNeuralNetworkFitter):
    """
    This class implements a Residual Neural Network (ResNet) for fitting spectra.
    """

    def __init__(self, number_of_components: int, number_of_channels: int) -> None:
        """
        Constructs a ResNetFitter object.

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

        self.input_layer = Sequential(
            Conv1d(1, 32, kernel_size=7, padding=3),
            BatchNorm1d(32),
            ReLU()
        )

        self.residual_layers = Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            MaxPool1d(2),
            ResidualBlock(32, 3),
            ResidualBlock(32, 3),
            MaxPool1d(2)
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(32 * (self.number_of_channels // 4), self.number_of_channels),
            ReLU(),
            Linear(self.number_of_channels, 3 * number_of_components)
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
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.fully_connected_layers(x)
        x = x.reshape(x.shape[0], self.number_of_components, 3)
        return x
