import numpy as np


def mean_squared_error(fitted_params: np.ndarray, true_params: np.ndarray) -> float:
    """
    Computes the mean squared error (MSE) between the fitted Gaussian parameters and the true parameters.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted Gaussian parameters (e.g., [mean, std_dev]).
    true_params : np.ndarray
        The true Gaussian parameters (e.g., [mean, std_dev]).

    Returns
    -------
    float
        The mean squared error between the parameters.
    """
    return np.mean((fitted_params - true_params) ** 2)
