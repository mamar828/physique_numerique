import numpy as np
from typing import Protocol, runtime_checkable
from graphinglib import Curve, Scatter, Line


@runtime_checkable
class CustomModel(Protocol):
    """
    This class implements a base CustomModel class used for type hinting.
    """
    number_of_parameters: int
    
    def __call__(self, x: np.ndarray, *args: float) -> np.ndarray:
        """
        Evaluates the CustomModel object at a given x. The average parameters are used to evaluate the model.

        Parameters
        ----------
        x : np.ndarray
            The x value to evaluate the CustomModel object.
        args: float
            The parameters of the model.

        Returns
        -------
        np.ndarray
            The evaluated CustomModel object at x.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Gives a string representation of the CustomModel object.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """
        Gives the number of parameters of the CustomModel object.
        """
        raise NotImplementedError

    @property
    def avg_mean(self) -> float:
        """
        Gives the average mean value.
        """
        raise NotImplementedError

    def get_plot(self, number_of_channels: int, color: str=None) -> list[Curve | Scatter | Line]:
        """
        Gives the plot of the CustomModel object. The average parameters are used to plot the model. The plot
        features errorbars representing the parameters's standard deviation.

        Parameters
        ----------
        number_of_channels : int
            The number of channels to plot the CustomModel object.

        color : str, optional
            The color of the plot.

        Returns
        -------
        list[Curve | Scatter | Line]
            The plot of the CustomModel object.
        """
        raise NotImplementedError

    def evaluate(self, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the model at the given x, n times. If the model has variable parameters, the parameters are randomly
        chosen from a random distribution whose standard deviation is one third of the difference between a bound and
        the parameter's average (one sixth of the difference between the upper and lower bounds). The number of
        evaluations is the same as the number of rows in x.

        Parameters
        ----------
        x : np.ndarray
            The x values to evaluate the models at. This is a 1D array with shape (m,) where m is the number of
            channels.
        n : int
            The number of times to evaluate the model.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The evaluated models at x and the corresponding parameters used for generation. The first array has shape
            (n,m) where n is the number of evaluations and m is the number of channels. The second array has shape (n,k)
            where n is the number of evaluations and k is the number of parameters the model has.
        """
        raise NotImplementedError
