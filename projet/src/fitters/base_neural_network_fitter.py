import numpy as np
import torch
from torch.nn import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from typing import Self
from copy import deepcopy
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
            train_loader: DataLoader,
            validation_loader: DataLoader=None,
            n_epochs: int=10,
            learning_rate: float=1e-3,
            learning_rate_decay: float=0.99
    ) -> tuple[list[float], list[float]]:
        """
        Trains the model using the given data loader.

        Parameters
        ----------
        train_loader : DataLoader
            Data loader used for training data.
        validation_loader : DataLoader, default=None
            Data loader used for validation data. If None, no validation is performed. This allows the model to keep
            only the best state it has achieved during training according to the validation set.
        n_epochs : int, default=10
            Number of epochs for training. After each epoch, the learning rate is adjusted using a ExponentialLR
            scheduler.
        learning_rate : float, default=1e-3
            Base learning rate with which to initialize the Adam optimizer.
        learning_rate_decay : float, default=0.99
            Learning rate decay with which to initialize the ExponentialLR learning rate scheduler, given as the gamma
            parameter.

        Returns
        -------
        tuple[list[float], list[float]]
            The training and validation losses for each epoch. If no validation loader is provided, the second list will
            be empty. This can be used to plot the training and validation losses over epochs.
        """
        self.train()
        self.to(self.DEVICE)

        criterion = MSELoss()           # a mean squared error loss is used for training
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=learning_rate_decay)

        start_epochs = time()
        epoch_train_losses = []
        epoch_validation_losses = []
        best_validation_loss = float('inf')
        best_model_state = None

        for i in range(1, n_epochs + 1):
            train_losses = []       # the train losses are stored for convenience
            for spectrum, params in tqdm(train_loader, f"Epoch {i}", len(train_loader), colour="#f39811", unit="batch"):
                spectrum = spectrum.to(self.DEVICE, non_blocking=True)
                params = params.to(self.DEVICE, non_blocking=True)

                optimizer.zero_grad()

                predictions = self(spectrum)
                loss = criterion(predictions, params)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
            
            lr_scheduler.step()
            epoch_train_loss = np.mean(train_losses)
            epoch_train_losses.append(epoch_train_loss)
            print(f"{'Train loss':>20}: {epoch_train_loss:.4f}")

            if validation_loader is not None:
                epoch_validation_loss = self.compute_loss(validation_loader, criterion)
                epoch_validation_losses.append(epoch_validation_loss)
                print(f"{'Validation loss':>20}: {epoch_validation_loss:.4f}")

                # Save the model state if validation loss improves
                if epoch_validation_loss < best_validation_loss:
                    best_validation_loss = epoch_validation_loss
                    best_model_state = deepcopy(self.state_dict())

        # Restore the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        print(f"{C.LIGHT_GREEN}{'-'*29}\nTRAINING FINISHED IN {time()-start_epochs:7.2f}s\n{'-'*29}{C.END}")
        return epoch_train_losses, epoch_validation_losses

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Computes the predictions of the model on the given data loader.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader containing the data to compute the predictions. A DataLoader with a very large batch size is
            recommended to speed up the process.

        Returns
        -------
        torch.Tensor
            The predictions of the model on the given data loader.
        """
        model_was_in_training = self.training
        self.eval()
        all_predictions = []

        # Use torch.no_grad() to disable gradient computation for faster inference
        with torch.no_grad():
            for spectrum, _ in tqdm(data_loader, f"Predicting", len(data_loader), colour="RED", unit="batch"):
                spectrum = spectrum.to(self.DEVICE, non_blocking=True)  # Use non_blocking for faster data transfer
                predictions = self(spectrum)
                all_predictions.append(predictions.cpu())

        if model_was_in_training:
            self.train()
        # Concatenate all predictions at once for better performance
        return torch.cat(all_predictions, dim=0)

    def compute_loss(
            self,
            data_loader: DataLoader,
            criterion: Module=MSELoss()
    ) -> float:
        """
        Computes the loss of the model on the given data loader. This method is identical to the
        score.mean_squared_error function, but it uses the model to compute the predictions.

        Parameters
        ----------
        data_loader : DataLoader
            DataLoader containing the data to compute the loss. A DataLoader with a very large batch size is recommended
            to speed up the process.
        criterion : Module, default=MSELoss()
            Loss function to use for computing the loss.

        Returns
        -------
        float
            The loss of the model on the given data loader.
        """
        model_was_in_training = self.training
        self.eval()
        all_losses = []

        # Loop on every batch to add all losses
        with torch.no_grad():
            for spectrum, params in data_loader:
                spectrum = spectrum.to(self.DEVICE)
                params = params.to(self.DEVICE)

                predictions = self(spectrum)
                loss = criterion(predictions, params)
                all_losses.append(loss.item())

        if model_was_in_training:
            self.train()
        return np.mean(all_losses)

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
