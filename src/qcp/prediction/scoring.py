# Standard library imports
import math

# Third-party imports
import numpy as np

from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.special import logsumexp

# Score functions

def euclidean_distance(calibration_point, frequency_dict):
    """
    Calculate the Euclidean distance between the calibration point and the expected value 
    of a distribution given by a frequency dictionary.
    
    Parameters:
        calibration_point (float): The reference value.
        frequency_dict (dict): A dictionary with values as keys and their frequencies as values.
    
    Returns:
        float: The Euclidean distance between the calibration point and the expected value 
               of the distribution.
    """

    total_count = sum(frequency_dict.values())
    expected_value = sum(value * (freq / total_count) for value, freq in frequency_dict.items())

    return np.sqrt((expected_value - calibration_point) ** 2)


def nearest_neighbours(calibration_point, frequency_dict, k):
    """
    Compute the k-th nearest Euclidean distance from a calibration point 
    to a set of values provided in a frequency distribution.
    
    Parameters:
        calibration_point (float): The reference value.
        frequency_dict (dict): A dictionary where keys are values and values 
                               are the frequencies (counts) of those values.
        k (int): The k-th nearest neighbor (1-indexed) to return.
    
    Returns:
        float: The k-th smallest Euclidean distance between the calibration point 
               and the values from the frequency dictionary.
    """
    euclidean_distances = np.array([])

    for real_value, frequency in frequency_dict.items():
        euclidean_distances = np.append(euclidean_distances, [real_value] * frequency)

    euclidean_distances = np.sqrt((euclidean_distances - calibration_point) ** 2)

    # Sort the distances in ascending order
    sorted_distances = np.sort(euclidean_distances, kind="quicksort")

    # Return the k-th nearest Euclidean distance
    return sorted_distances[k-1]

def histogram(calibration_point, frequency_dict, M, tau):
    """
    Compute a weighted histogram score for a calibration point using a frequency distribution.
    
    Parameters:
        calibration_point (float): The reference value to score.
        frequency_dict (dict): A dictionary mapping binary counts to their frequency counts.
        M (int): The number of elements (or iterations) to consider.
        tau (float): The decoherence time parameter.

    Returns:
        float: The histogram score computed as M divided by the sum of the weighted and counts.
    """
    counts = []

    for key, count in frequency_dict.items():
        counts.extend([key] * count)

    weights = [
        (math.exp() ** ((-m) / tau)) * (calibration_point == counts[m])
        for m in range(M)
    ]
    weights_sum = np.sum(weights)

    return M / weights_sum

def euclidean_distance(calibration_point, frequency_dict):
    """
    Calculate the Euclidean distance between the calibration point and the expected value 
    of a distribution given by a frequency dictionary.
    
    Parameters:
        calibration_point (float): The reference value.
        frequency_dict (dict): A dictionary with values as keys and their frequencies as values.
    
    Returns:
        float: The Euclidean distance between the calibration point and the expected value 
               of the distribution.
    """

    total_count = sum(frequency_dict.values())
    expected_value = sum(value * (freq / total_count) for value, freq in frequency_dict.items())

    return np.sqrt((expected_value - calibration_point) ** 2)
