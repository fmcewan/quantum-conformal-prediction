# Standard library imports
import numpy as np

# Local application imports
from distributions.combined_normals import CombinedNormals
from distributions.heteroscedastic import HeteroscedasticData
from distributions.sinusoidal import SinusoidalData
from distributions.skewed_normal import SkewedNormal
from distributions.normal import Normal
from distributions.random_gibbs_states import RandomGibbsStates

def create_distribution(data_config):
    """
    Create and return a distribution object based on the provided configuration.

    The function inspects the 'distribution' key in the data configuration dictionary
    and returns an instance of the corresponding distribution class with the given parameters.

    Parameters:
        data_config (dict): A dictionary containing configuration parameters for the distribution.
            Expected keys vary by distribution type, e.g. "combined_normals" requires 'component_means' and 'component_stds'

    Returns:
        An instance of a distribution object corresponding to the specified distribution type.
    """

    distribution_type = data_config["distribution"]

    if distribution_type == "combined_normals":
        return CombinedNormals(data_config["component_means"], data_config["component_stds"])
    elif distribution_type == "normal":
        return Normal(data_config['mean'], data_config['scale'])
    elif distribution_type == "sinusoidal":
        return SinusoidalData(data_config["x_range"])
    elif distribution_type == "heteroscedastic":
        return HeteroscedasticData(data_config["x_range"])
    elif distribution_type == "skewed_normal":
        return SkewedNormal(data_config["loc"], data_config["scale"], skew=data_config["skew"])
    elif distribution_type == "classification":
        return RandomGibbsStates(data_config["num_classes"], data_config["num_features"], data_config["dimension"], data_config["density"], data_config["temperature"]);
    else:
        raise ValueError(f"Unknown distribution type: {distribution_type}")

def event_probability(dist, event, x=None):
    """
    Compute the probability of an event, optionally conditioned on a value x.

    Parameters:
        dist: A distribution object with methods 'cdf' and 'cdf_given_x'.
        event (list of tuple): A list of (lower, upper) intervals defining the event.
        x (optional): The value at which to condition the probability. If None, the unconditional
                      probability is computed.

    Returns:
        float: The total probability of the event.
    """

    if x is None:
        return np.sum(
            [dist.cdf(interval[1]) - dist.cdf(interval[0]) for interval in event]
        )

    return np.sum(
        [dist.cdf_given_x(interval[1], x) - dist.cdf_given_x(interval[0], x) for interval in event]
    )
