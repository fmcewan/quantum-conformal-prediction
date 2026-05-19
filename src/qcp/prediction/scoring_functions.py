# Standard library imports
import math

# Third-party imports
import numpy as np

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
    
    distances = []
    for value, frequency in frequency_dict.items():
        distances.extend([abs(value - calibration_point)] * frequency)
    
    return sorted(distances)[k-1]

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
    
    # TODO: Test and implement function in pipeline
    
    counts = []

    for key, count in frequency_dict.items():
        counts.extend([key] * count)

    weights = [(math.exp((-m)/tau) ** ((-m)/tau)) * (calibration_point == counts[m]) for m in range(M)]
    weights_sum = np.sum(weights)

    return M / weights_sum

