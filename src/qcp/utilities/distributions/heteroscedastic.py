import numpy as np
from scipy.stats import norm, rv_continuous
import torch
import math
from distributions.combined_normals import CombinedNormals
from scipy.stats import norm

class HeteroscedasticData():
    """
    Generates heteroscedastic data with varying mean and standard deviation based on x.
    """

    def __init__(self, x_range):
        """
        Initialize with range of x-values.
        """
        super().__init__()
        self.min_x = x_range[0]
        self.max_x = x_range[1]

    def component_mean(self, x):
        """
        Compute the mean based on x.
        """
        def single_value_mean(x_val):
            return 3*math.sin(2*x_val)/(4*x_val**2) - 3*math.cos(2*x_val)/(2*x_val) + (x_val-6)/12

        return np.vectorize(single_value_mean)(x) if not np.isscalar(x) else single_value_mean(x)

    def component_std(self, x):
        """
        Compute the standard deviation based on x.
        """
        return 0.015 + 0.02 * x
    
    def pdf(self, x, y):
        """
        Compute the PDF for x.
        """
        component_mean = self.component_mean(x)
        component_std = self.component_std(x)
        dist = norm(loc=component_mean, scale=component_std)
        return dist.pdf(y)

    def cdf_given_x(self, y, x):
        """
        Compute the CDF for y given x.
        """
        component_mean = self.component_mean(x)
        component_std = self.component_std(x)
        dist = norm(loc=component_mean, scale=component_std)
        return dist.cdf(y)
    
    def rvs(self, size=1, random_state=None):
        """
        Generate random samples from the distribution.
        """
        x_points = np.random.uniform(self.min_x, self.max_x, size)
        y_points = np.zeros(size)

        for i, x in enumerate(x_points):
            mean = self.component_mean(x)
            std = self.component_std(x)
            y_points[i] = np.random.normal(loc=mean, scale=std, size=1)

        return x_points, y_points

    def get_conditional_hdr(self, x, percentage):
        mean = self.component_mean(x)
        std = self.component_std(x)
        dist = norm(loc=mean, scale=std)
        lower_bound, upper_bound = dist.interval(percentage)
        return [[lower_bound, upper_bound]]
