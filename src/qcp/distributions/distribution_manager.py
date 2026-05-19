# Standard library imports
import numpy as np

# Local application imports
from qcp.distributions.combined_normals import CombinedNormals
from qcp.distributions.heteroscedastic import HeteroscedasticData
from qcp.distributions.sinusoidal import SinusoidalData
from qcp.distributions.skewed_normal import SkewedNormal
from qcp.distributions.normal import Normal
from qcp.distributions.random_gibbs_states import RandomGibbsStates

def create_distribution(data_configuration):

    distribution_type = data_configuration["distribution"]

    if distribution_type == "combined_normals":
        return CombinedNormals(data_configuration["component_means"], data_configuration["component_stds"])
    elif distribution_type == "normal":
        return Normal(data_configuration['mean'], data_configuration['scale'])
    elif distribution_type == "sinusoidal":
        return SinusoidalData(data_configuration["x_range"])
    elif distribution_type == "heteroscedastic":
        return HeteroscedasticData(data_configuration["x_range"])
    elif distribution_type == "skewed_normal":
        return SkewedNormal(data_configuration["loc"], data_configuration["scale"], skew=data_configuration["skew"])
    elif distribution_type == "classification":
        return RandomGibbsStates(data_configuration["num_classes"], data_configuration["num_features"], data_configuration["dimension"], data_configuration["density"], data_configuration["temperature"]);
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def event_probability(distribution, event, x=None):

    if x is None:
        return np.sum([distribution.cdf(interval[1]) - distribution.cdf(interval[0]) for interval in event])

    return np.sum([distribution.cdf_given_x(interval[1], x) - distribution.cdf_given_x(interval[0], x) for interval in event])
