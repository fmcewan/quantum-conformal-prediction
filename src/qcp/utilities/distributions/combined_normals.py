# Standard library imports
import numpy as np

# Third-party imports
from scipy.stats import norm, rv_continuous

# Local application imports
from utils.graphing_tricks import calculate_ranges

class CombinedNormals(rv_continuous):
    
    def __init__(self, component_means, component_stds):
        """
        Initialize the mixture distribution.

        Parameters:
            component_means (list or array-like): Means of the normal distributions.
            component_stds (list or array-like): Standard deviations of the normal distributions.
        """
        super().__init__()
        if len(component_means) != len(component_stds) or len(component_means) == 0:
            raise ValueError("Means and standard deviations must be equal and non-empty.")
        self.component_means = component_means
        self.component_stds = component_stds
        self.n_modes = len(component_means)


    def _pdf(self, x):
        """
        Compute the probability density function (PDF) of the mixture at x.

        Parameters:
            x (float or array-like): The value(s) at which to compute the PDF.

        Returns:
            float or array-like: The PDF evaluated at x.
        """
        pdf_vals = np.array([
            norm.pdf(x, self.component_means[i], self.component_stds[i])
            for i in range(self.n_modes)
        ])
        return np.mean(pdf_vals, axis=0)


    def _cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) of the mixture at x.

        Parameters:
            x (float or array-like): The value(s) at which to compute the CDF.

        Returns:
            float or array-like: The CDF evaluated at x.
        """
        return np.mean([
            norm.cdf(x, self.component_means[i], self.component_stds[i])
            for i in range(self.n_modes)
        ])

    
    def _rvs(self, size=1, random_state=None):
        """
        Generate random samples from the mixture distribution.

        Parameters:
            size (int, optional): The number of random samples to generate (default is 1).

        Returns:
            ndarray: Random samples drawn from the mixture distribution.
        """
        choices = np.random.choice(self.n_modes, size=size)
        means = np.take(self.component_means, choices)
        stds = np.take(self.component_stds, choices)
        return np.random.normal(means, stds)

    
    def pdf_quantile_threshold(self, quantile):
        """
        Find the PDF value threshold such that the total probability mass of points 
        with a PDF value greater than or equal to the threshold equals the given quantile.

        Parameters:
            quantile (float): The desired quantile of the distribution (e.g., 0.95 for the top 95%).

        Returns:
            float: The PDF value corresponding to the desired quantile mass.
        """
        x_vals = np.linspace(-2, 2, 1000)
        pdf_vals = self._pdf(x_vals)
        sorted_pdf_vals = np.sort(pdf_vals)[::-1]
        cumulative_prob = np.cumsum(sorted_pdf_vals) * np.diff(x_vals)[0]
        threshold_index = np.where(cumulative_prob >= 1 - quantile)[0][0]
        return sorted_pdf_vals[threshold_index]
    
    def get_hdr(self, mass, x_range=[-3, 3], granularity=1000):
        """
        Find the high density regions (HDR) of the distribution for a specified mass.

        Parameters:
            mass (float): The desired mass (e.g., 0.95 for 95% HDR).
            range (list or tuple, optional): The (min, max) range over which to evaluate the PDF (default is [-3, 3]).
            granularity (int, optional): The number of points to evaluate within the specified range (default is 1000).

        Returns:
            list: A list of ranges representing the high density regions.
        """
        test_points = np.linspace(x_range[0], x_range[1], granularity)
        pdfs = self.pdf(test_points)
        quantile_pdf = self.pdf_quantile_threshold(1 - mass)
        true_quantile_set = test_points[pdfs > quantile_pdf]
        return calculate_ranges(true_quantile_set, x_range[0], x_range[1], granularity)
