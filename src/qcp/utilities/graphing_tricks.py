import numpy as np

def calculate_ranges(prediction_set, min_val, max_val, n):
    # if len(prediction_set==0):
    #     return []
    # Calculate the standard distance between points
    standard_distance = (max_val - min_val) / (n - 1)
    half_distance = standard_distance / 2  # Half the standard distance

    # Create an array of ranges
    ranges = []
    start = prediction_set[0]

    for i in range(1, len(prediction_set)):
        # Check if the current point is not contiguous with the previous point
        if prediction_set[i] - prediction_set[i - 1] >= standard_distance*1.5: # 1.5 for tolerance
            # Add the range adjusted by half the standard distance
            ranges.append((start - half_distance, prediction_set[i - 1] + half_distance))
            start = prediction_set[i]

    # Append the final range, also adjusted by half the standard distance
    ranges.append((start - half_distance, prediction_set[-1] + half_distance))

    return ranges

