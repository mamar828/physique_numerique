import numpy as np
from astropy.modeling.models import Gaussian1D
from graphinglib import Curve, Scatter, Line

from projet.src.models.custom_model import CustomModel


class CustomGaussian(CustomModel):
    """
    This class implements a custom Gaussian model, which can have lower and upper limits on each of its parameters.
    """
    number_of_parameters: int = 3

    def __init__(
            self,
            amplitude: float | tuple[float, float],
            mean: float | tuple[float, float],
            stddev: float | tuple[float, float],
    ):
        """
        Initializes a CustomGaussian object.

        Parameters
        ----------
        amplitude : float | tuple[float, float]
            The amplitude of the Gaussian. If a tuple of floats is given, the first element is the lower limit and the
            second element is the upper limit.
        mean : float | tuple[float, float]
            The mean of the Gaussian. If a tuple of floats is given, the first element is the lower limit and the
            second element is the upper limit.
        stddev : float | tuple[float, float]
            The standard deviation of the Gaussian. If a tuple of floats is given, the first element is the lower limit
            and the second element is the upper limit.
        """
        self.amplitude = amplitude
        self.mean = mean
        self.stddev = stddev

    def __call__(
            self,
            x: np.ndarray,
            amplitude: float=None,
            mean: float=None,
            stddev: float=None
    ) -> np.ndarray:
        """
        Evaluates the CustomGaussian object at a given x.

        Parameters
        ----------
        x : np.ndarray
            The x value to evaluate the CustomGaussian object.
        amplitude : float, optional
            The amplitude of the Gaussian. Defaults to the average amplitude if None.
        mean : float, optional
            The mean of the Gaussian. Defaults to the average mean if None.
        stddev : float, optional
            The standard deviation of the Gaussian. Defaults to the average standard deviation if None.

        Returns
        -------
        np.ndarray
            The evaluated CustomGaussian object at x.
        """
        return Gaussian1D(
            amplitude if amplitude is not None else self.avg_amplitude,
            mean if mean is not None else self.avg_mean,
            stddev if stddev is not None else self.avg_stddev
        )(x)

    def __str__(self) -> str:
        """
        Gives a string representation of the CustomGaussian object.
        """
        return f"CustomGaussian(amplitude={self.amplitude}, mean={self.mean}, stddev={self.stddev})"
    
    def __len__(self):
        """
        Gives the number of parameters of the CustomGaussian object.
        """
        return 3

    @property
    def avg_amplitude(self) -> float:
        """
        Gives the average amplitude value.
        """
        return self.amplitude if isinstance(self.amplitude, (float, int)) else sum(self.amplitude) / 2

    @property
    def avg_mean(self) -> float:
        """
        Gives the average mean value.
        """
        return self.mean if isinstance(self.mean, (float, int)) else sum(self.mean) / 2

    @property
    def avg_stddev(self) -> float:
        """
        Gives the average stddev value.
        """
        return self.stddev if isinstance(self.stddev, (float, int)) else sum(self.stddev) / 2

    def get_plot(self, number_of_channels: int, color: str=None) -> list[Curve | Scatter | Line]:
        """
        Gives the plot of the CustomGaussian object. The average parameters are used to plot the Gaussian. The plot
        features errorbars representing the parameters's standard deviation.

        Parameters
        ----------
        number_of_channels : int
            The number of channels to plot the CustomGaussian object.

        color : str, optional
            The color of the plot.

        Returns
        -------
        list[Curve | Scatter | Line]
            The plot of the CustomGaussian object.
        """
        x = np.arange(number_of_channels) + 1       # first channel starts at 1

        gaussian = Curve(x, self(x), line_width=2, line_style=":", color=color)
        peak = Scatter(self.avg_mean, self.avg_amplitude, face_color=color, marker_style="o", marker_size=50)
        peak.add_errorbars(
            x_error=np.diff(self.mean) / 2 if isinstance(self.mean, tuple) else None,
            y_error=np.diff(self.amplitude) / 2 if isinstance(self.amplitude, tuple) else None,
        )
        plottables = [gaussian, peak]
        if isinstance(self.stddev, tuple):
            half_width = np.sqrt(2 * np.log(2)) * np.array(self.stddev)
            plottables.append(Line(
                    [self.avg_mean + half_width[0], self.avg_amplitude/2], 
                    [self.avg_mean + half_width[1], self.avg_amplitude/2], 
                    color=color,
                    capped_line=True,
                    cap_width=0.3,
                    width=1
            ))

        return plottables

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
            (n,m) where n is the number of evaluations and m is the number of channels. The second array has shape (n,3)
            where n is the number of evaluations and the columns are the amplitude, mean, and standard deviation
            of each gaussian.
        """
        params = np.random.normal(
            loc=[self.avg_amplitude, self.avg_mean, self.avg_stddev],
            scale=[
                (self.amplitude[1] - self.amplitude[0]) / 6 if isinstance(self.amplitude, tuple) else 0,
                (self.mean[1] - self.mean[0]) / 6 if isinstance(self.mean, tuple) else 0,
                (self.stddev[1] - self.stddev[0]) / 6 if isinstance(self.stddev, tuple) else 0,
            ],
            size=(n, 3)
        )
        # Clip all negative values to 0
        params = np.clip(params, a_min=0, a_max=None)

        amplitude, mean, stddev = params.T[:,:,None]

        # Add a small epsilon to stddev to avoid division by zero
        return amplitude * np.exp(- (x - mean) ** 2 / (2 * (stddev + 1e-10) ** 2)), params
