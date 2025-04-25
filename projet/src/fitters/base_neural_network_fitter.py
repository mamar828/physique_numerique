import numpy as np
import torch
from torch.nn import *
from torch.utils.data import DataLoader
import graphinglib as gl
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from time import time
from typing import Self, Generator
from eztcolors import Colors as C

from projet.src.tools.messaging import notify_function_end


class BaseNeuralNetworkFitter(torch.nn.Module):
    """
    This class implements basic convenient methods for using any neural network fitter. The machine learning models
    should derive from this base class.
    """

    def __init__(self):
        """
        Constructs a BaseNeuralNetworkFitter object.
        """
        super().__init__()
        self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"{C.GREEN}{self.__class__.__name__} initialized on {self.DEVICE}{C.END}")

    @classmethod
    def load(cls, filename: str) -> Self:
        """
        Loads a model from a file containing its state dict and initialization parameters.

        Parameters
        ----------
        filename : str
            Filename of the file containing the model's information.

        Returns
        -------
        Self
            The loaded model.
        """
        DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model_data = torch.load(filename, map_location=DEVICE, weights_only=True)
        model = cls(**model_data['init_params'])  # Pass saved init parameters to the constructor
        model.load_state_dict(model_data['state_dict'])
        return model

    def save(self, filename: str):
        """
        Saves the model's state dict and initialization parameters to a file.

        Parameters
        ----------
        filename : str
            Filename of the file in which the model's information must be saved.
        """
        param_names = self.__init__.__code__.co_varnames[1:]  # Get the names of the parameters in the constructor
        param_values = [getattr(self, name) for name in param_names]  # Get the values of the parameters
        init_params = dict(zip(param_names, param_values))  # Create a dictionary of the parameters
        torch.save({
            "state_dict": self.state_dict(),
            "init_params": init_params
        }, filename)

    @notify_function_end
    def training_loop(
            self,
            train_loader : DataLoader,
            n_epochs: int=10,
            learning_rate: float=1e-3,
            learning_rate_decay: float=0.99
    ) -> None:
        """
        Trains the model using the given data loader.

        Parameters
        ----------
        train_loader : DataLoader
            Data loader used for training data.
        n_epochs : int, default=10
            Number of epochs for training. After each epoch, the learning rate is adjusted using a ExponentialLR
            scheduler.
        learning_rate : float, default=1e-3
            Base learning rate with which to initialize the Adam optimizer.
        learning_rate_decay : float, default=0.99
            Learning rate decay with which to initialize the ExponentialLR learning rate scheduler, given as the gamma
            parameter.
        """
        self.train()

        criterion = MSELoss()           # a mean squared error loss is used for training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)
        self.to(self.DEVICE)

        start_epochs = time()
        for i in range(1, n_epochs+1):
            train_losses = []       # the train losses are stored for convenience
            for spectrum, params in tqdm(train_loader, f"Epoch {i}", len(train_loader), colour="#f39811", unit="batch"):
                spectrum = spectrum.to(self.DEVICE)
                params = params.to(self.DEVICE)

                optimizer.zero_grad()

                predictions = self(spectrum)
                loss = criterion(predictions, params)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
            
            lr_scheduler.step()
            print(f"Epoch {i} loss: {np.mean(train_losses):.4f}")

        torch.cuda.empty_cache()
        print(f"{C.LIGHT_GREEN}{'-'*29}\nTRAINING FINISHED IN {time()-start_epochs:7.2f}s\n{'-'*29}")

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Computes the predictions of the model on the given data loader.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader containing the data to compute the predictions.

        Returns
        -------
        torch.Tensor
            The predictions of the model on the given data loader.
        """
        self.eval()
        self.to(self.DEVICE)
        all_predictions = []

        # Loop on every batch to add all predictions and targets
        with torch.no_grad():
            for spectrum, _ in tqdm(data_loader, f"Predicting", len(data_loader), colour="RED", unit="batch"):
                spectrum = spectrum.to(self.DEVICE)

                # logits = self(images)
                # predictions = torch.sigmoid(logits) > 0.5
                predictions = self(spectrum)
                all_predictions.append(predictions)

        all_predictions = torch.cat(all_predictions)

        # Ensure predictions and targets are of the same dtype and device
        all_predictions = all_predictions.to(self.DEVICE)
        return all_predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model. This method should be implemented in the derived class.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, number_of_channels, number_of_channels).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, number_of_components, 3). For each component, the output contains its
            amplitude, mean and standard deviation.
        """
        raise NotImplementedError
