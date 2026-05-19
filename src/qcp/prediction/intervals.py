import numpy as np

def points_to_intervals(prediction_set: np.ndarray, min_val: float, max_val: float, n: int) -> list[tuple]:
    
    """
    Convert a sorted array of discrete prediction points into a list of continuous intervals.

    Each contiguous run of points is merged into a single interval, extended by half
    the standard grid spacing on each side.

    Parameters:
        prediction_set: Sorted array of predicted y values.
        min_val: Minimum value of the output range.
        max_val: Maximum value of the output range.
        n: Number of grid points used to generate the prediction set.

    Returns:
        A list of (lower, upper) tuples representing the prediction intervals.
    
    Raises:
        IndexError: If prediction_set is empty.
    """
    
    standard_distance = (max_val - min_val) / (n - 1)
    half_distance = standard_distance / 2

    ranges = []
    start = prediction_set[0]

    for i in range(1, len(prediction_set)):
        if prediction_set[i] - prediction_set[i - 1] >= standard_distance * 1.5:
            ranges.append((start - half_distance, prediction_set[i - 1] + half_distance))
            start = prediction_set[i]

    ranges.append((start - half_distance, prediction_set[-1] + half_distance))
    
    return ranges
