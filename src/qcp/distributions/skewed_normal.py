import numpy as np
from scipy.stats import skewnorm, rv_continuous
from qcp.utilities.graphing_tricks import calculate_ranges


class SkewedNormal(rv_continuous):
    """
    Represents a skewed normal distribution with specified location, scale, and skewness.
    """


    def __init__(self, loc, scale, skew):
        """
        Initialize the skewed normal distribution.

        Parameters:
        ----------
        loc : float
            The location (mean) of the distribution.
        scale : float
            The scale (standard deviation) of the distribution.
        skew : float
            The skewness of the distribution (positive for right skew, negative for left skew).
        """
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.skew = skew

    def _pdf(self, x):
        """
        Compute the PDF of the skewed normal distribution at x.
        """
        return skewnorm.pdf(x, self.skew, loc=self.loc, scale=self.scale)

    def _cdf(self, x):
        """
        Compute the CDF of the skewed normal distribution at x.
        """
        return skewnorm.cdf(x, self.skew, loc=self.loc, scale=self.scale)

    def _rvs(self, size=1, random_state=None):
        """
        Generate random samples from the skewed normal distribution.
        """
        return skewnorm.rvs(self.skew, loc=self.loc, scale=self.scale, size=size, random_state=random_state)

    def pdf_quantile_threshold(self, quantile):
        """
        Find the PDF value threshold such that the total probability mass
        of points with a PDF value greater than or equal to the threshold
        equals the given quantile.
        
        Parameters:
        ----------
        quantile : float
            The desired quantile of the distribution (e.g., 0.95 for the top 95%).

        Returns:
        --------
        threshold : float
            The PDF value corresponding to the desired quantile mass.
        """
        x_vals = np.linspace(self.loc - 4 * self.scale, self.loc + 4 * self.scale, 1000)

        # Compute the PDF values for the given x-values
        pdf_vals = self._pdf(x_vals)
        
        # Sort the PDF values in descending order
        sorted_pdf_vals = np.sort(pdf_vals)[::-1]
        
        # Calculate cumulative density from the sorted PDF values
        cumulative_prob = np.cumsum(sorted_pdf_vals) * np.diff(x_vals)[0]  # Assumes uniform spacing in x_vals
        
        # Find the index where cumulative probability reaches or exceeds the quantile
        threshold_index = np.where(cumulative_prob >= 1 - quantile)[0][0]
        
        # Return the PDF value at that index
        return sorted_pdf_vals[threshold_index]

    def get_hdr(self, mass, range=None, granularity=1000):
        """
        Find the high-density regions of the distribution given a specified mass.

        Parameters:
        ----------
        mass : float
            The desired mass for the HDR (e.g., 0.95 for the 95% HDR).
        range : list or tuple, optional
            The range over which to compute the HDR [min, max]. Default is ±4 standard deviations.
        granularity : int
            The number of points used for evaluating the PDF.

        Returns:
        --------
        list of tuples
            Intervals representing the HDR.
        """
        if range is None:
            range = [self.loc - 4 * self.scale, self.loc + 4 * self.scale]

        test_points = np.linspace(range[0], range[1], granularity)
        pdfs = self._pdf(test_points)
        quantile_pdf = self.pdf_quantile_threshold(1 - mass)
        true_quantile_set = test_points[pdfs > quantile_pdf]

        return calculate_ranges(true_quantile_set, range[0], range[1], granularity)
