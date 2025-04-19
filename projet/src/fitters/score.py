import numpy as np
from sklearn.metrics import r2_score

from projet.src.data_structures.spectrum_data_array import SpectrumDataArray


def mean_squared_error(fitted_params: np.ndarray, true_params: np.ndarray) -> float:
    """
    Computes the mean squared error (MSE) between the fitted Gaussian parameters and the true parameters.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted Gaussian parameters (e.g., [mean, std_dev]). The shape should be the same as true_params.
    true_params : np.ndarray
        The true Gaussian parameters (e.g., [mean, std_dev]). The shape should be the same as fitted_params.

    Returns
    -------
    float
        The mean squared error between the parameters.
    """
    return np.mean((fitted_params - true_params) ** 2)

def mean_r2_score(fitted_params: np.ndarray, data_array: SpectrumDataArray) -> float:
    """
    Computes the coefficient of determination (R^2) between the fitted Gaussian parameters and the data. The mean
    r2_score value of each evaluation is used.

    Parameters
    ----------
    fitted_params : np.ndarray
        An array that contains the fitted parameters for each model. The shape is (n,j,k) where n is the number of
        evaluations, j is the number of models and k is the number of parameters per model. 
    data_array : SpectrumDataArray
        The data array for comparing the fitted parameters. This is used to compute the R^2 value.
    # spectrum_data : np.ndarray
    #     A size (n,m) numpy array containing n spectra with m channels each. This data is used to compute the R^2 value.

    Returns
    -------
    float
        The coefficient of determination between the parameters.
    """
    y_true = data_array.data
    y_pred = [data_array.spectrum(data_array.spectrum.x_values, fitted_params_i) for fitted_params_i in fitted_params]
    r2_scores = []
    for y_true_i, y_pred_i in zip(y_true, y_pred):
        r2_scores.append(r2_score(y_true_i, y_pred_i))
    
    return np.mean(r2_scores)
