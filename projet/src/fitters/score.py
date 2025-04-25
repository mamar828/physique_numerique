import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm

from projet.src.data_structures.spectrum_data_object import SpectrumDataObject
from projet.src.data_structures.spectrum_dataset import SpectrumDataset
from projet.src.models.custom_gaussian import CustomGaussian


def mean_squared_error(
        fitted_params: np.ndarray,
        true_params: np.ndarray
) -> float:
    """
    Computes the mean squared error (MSE) between the fitted parameters and the true parameters. If nan values are
    present in the fitted parameters, they are ignored in the computation. The MSE is calculated as the mean of the
    squared differences between the fitted and true parameters: MSE = 1/N * Σ (fitted_params - true_params)^2, where N
    is the number of parameters.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted parameters (e.g., [mean, std_dev]). The shape should be the same as true_params.
    true_params : np.ndarray
        The true parameters (e.g., [mean, std_dev]). The shape should be the same as fitted_params.

    Returns
    -------
    float
        The mean squared error between the parameters. It converges to 0 when the fitted parameters are close to the
        true parameters.
    """
    return np.nanmean((fitted_params - true_params) ** 2)

def custom_mean_squared_error(
        fitted_params: np.ndarray, 
        true_params: np.ndarray,
        eps: float = 1e-10
) -> float:
    """
    Computes a custom mean squared error (MSE) between the fitted parameters and the true parameters. If nan values are
    present in the fitted parameters, they are ignored in the computation. The CMSE is calculated as the mean of the
    squared relative error of each parameter: 
    CMSE = 1/N * Σ ((fitted_params - true_params) / true_params)^2, where N is the number of parameters.
    Warning: if the true_params are 0, the CMSE will be infinite. To avoid this, a small value (eps) is added to the
    true_params to prevent infinite values. However, this may lead to large values if the fitted_params are not as small
    as the true_params.

    Parameters
    ----------
    fitted_params : np.ndarray
        The fitted parameters (e.g., [mean, std_dev]). The shape should be the same as true_params.
    true_params : np.ndarray
        The true parameters (e.g., [mean, std_dev]). The shape should be the same as fitted_params.
    eps : float, default=1e-10
        A small value to avoid division by zero. This value is added to the true_params to prevent infinite values.

    Returns
    -------
    float
        The custom mean squared error between the parameters. It converges to 0 when the fitted parameters are close to
        the true parameters.
    """
    # The simplified CMSE equation is used
    return np.nanmean((fitted_params / (true_params+eps) - 1) ** 2)

def mean_r2_score(
        fitted_params: np.ndarray, 
        spectrum_data: SpectrumDataObject
) -> float:
    """
    Computes the coefficient of determination (R^2) between the fitted parameters and the data. The mean r2_score value
    of each evaluation is used.

    Parameters
    ----------
    fitted_params : np.ndarray
        An array that contains the fitted parameters for each model. The shape is (n,j,k) where n is the number of
        evaluations, j is the number of models and k is the number of parameters per model. 
    spectrum_data : SpectrumDataObject
        The spectrum data object for comparing the fitted parameters. This is used to compute the R^2 value. If a
        SpectrumDataset is given, the tensors are converted to numpy arrays.

    Returns
    -------
    float
        The coefficient of determination between the parameters. It converges to 1 when the fitted parameters are close
        to the spectrum data.
    """
    if isinstance(spectrum_data, SpectrumDataset):
        y_true = spectrum_data.data.squeeze(1)
        fitted_params = fitted_params.numpy()
    else:
        y_true = spectrum_data.data

    # If spectrum_data.spectrum contains only gaussians, the computation can be greatly accelerated
    if all(isinstance(model, CustomGaussian) for model in spectrum_data.spectrum.models):
        # Convert all parameters to 3D arrays so the operations are made for all spectra, for all models and for all
        # channels
        x_values = spectrum_data.spectrum.x_values[None,None,:]
        A = fitted_params[:,:,0][...,None]
        mu = fitted_params[:,:,1][...,None]
        sigma = fitted_params[:,:,2][...,None]

        gaussian_values = A * np.exp(-((x_values - mu) / sigma) ** 2)
        y_pred = np.sum(gaussian_values, axis=1)
        return r2_score(y_true, y_pred, multioutput="uniform_average")

    else:
        y_pred = (
            spectrum_data.spectrum(spectrum_data.spectrum.x_values, fitted_params_i) 
            for fitted_params_i in fitted_params
        )           # this creates a generator of the fitted parameters

        r2_scores = []
        for y_true_i, y_pred_i in tqdm(zip(y_true, y_pred), f"Computing R^2", len(y_true), colour="RED", unit="fit"):
            r2_scores.append(r2_score(y_true_i, y_pred_i))

        return np.mean(r2_scores)
