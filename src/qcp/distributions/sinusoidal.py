import numpy as np
from qcp.distributions.combined_normals import CombinedNormals
from scipy.optimize import brentq

class Sinusoidal:

    def __init__(self, x_range):
        """
        Initialize with the range of x-values.
        """
        self.min_x = x_range[0]
        self.max_x = x_range[1]

    def component_mean(self, x):
        """
        Compute the sinusoidal component mean for x.
        """
        return 0.5 * np.sin(0.8 * x) + 0.05 * x
    
    def component_std(self):
        return 0.05

    def pdf(self, y, x):
        """
        Compute the PDF for y given x.
        """
        component_mean = self.component_mean(x)
        component_std = self.component_std()
        dist = CombinedNormals([-component_mean, component_mean], [component_std, component_std])
        return dist.pdf(y)
    
    def cdf_given_x(self, y, x):
        """
        Compute the CDF for y given x.
        """
        component_mean = self.component_mean(x)
        component_std = self.component_std()
        dist = CombinedNormals([-component_mean, component_mean], [component_std, component_std])
        return dist.cdf(y)
    
    def ppf(self, x, quantile):
        """
        Compute the inverse CDF (quantile function) for x and a given quantile
        """
        def objective(y, x, quantile):
            return self.cdf(x, y) - quantile
        return brentq(objective, self.min_x, self.max_x, args=(x, quantile))

    def rvs(self, size):
        """
        Generate random samples from the distribution.
        """
        x_points = np.random.uniform(self.min_x, self.max_x, size=size)
        component_mean_values = self.component_mean(x_points)

        y_points = np.array([
            CombinedNormals([-mean, mean], [0.05, 0.05]).rvs()
            for mean in component_mean_values
        ])
        return (x_points, y_points)


    def get_conditional_hdr(self, x, percentage):
        mean = self.component_mean(x)
        std = self.component_std(x)
        dist = CombinedNormals([-mean, mean], [std, std])
        return dist.get_hdr(percentage)
