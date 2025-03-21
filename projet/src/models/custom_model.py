import numpy as np
from typing import Protocol, runtime_checkable


@runtime_checkable
class CustomModel(Protocol):
    """
    This class implements a base CustomModel class used for type hinting.
    """
    
    def evaluate_gaussian(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluates the CustomModel object at a given x.

        Parameters
        ----------
        x : np.ndarray
            The x value to evaluate the CustomModel object.

        Returns
        -------
        np.ndarray
            The evaluated CustomModel object at x.
        """
        raise NotImplementedError
