import pytest
import numpy as np

from qcp.prediction.intervals import points_to_intervals

def make_grid(min_val, max_val, n):
    return np.linspace(min_val, max_val, n)

def test_single_contiguous_range():
    
    grid = make_grid(0, 1, 11)  # points at 0, 0.1, 0.2, ..., 1.0
    result = points_to_intervals(grid, 0, 1, 11)
    
    assert len(result) == 1

def test_two_separate_ranges():
    
    grid = make_grid(0, 1, 11)
    prediction_set = np.concatenate([grid[:3], grid[8:]])
    result = points_to_intervals(prediction_set, 0, 1, 11)
   
    assert len(result) == 2

def test_ranges_cover_endpoints():
    
    grid = make_grid(0, 1, 11)
    result = points_to_intervals(grid, 0, 1, 11)
    half_distance = (1 - 0) / (11 - 1) / 2
    
    assert result[0][0] == pytest.approx(0 - half_distance)
    assert result[0][1] == pytest.approx(1 + half_distance)

def test_single_point():
    grid = make_grid(0, 1, 11)
    prediction_set = np.array([grid[5]])  # just the middle point
    result = points_to_intervals(prediction_set, 0, 1, 11)
    
    assert len(result) == 1
    
    half_distance = (1 - 0) / (11 - 1) / 2
    
    assert result[0][0] == pytest.approx(grid[5] - half_distance)
    assert result[0][1] == pytest.approx(grid[5] + half_distance)

def test_range_widths_are_positive():
    
    grid = make_grid(-5, 5, 100)
    prediction_set = grid[10:40]
    result = points_to_intervals(prediction_set, -5, 5, 100)
    
    for lower, upper in result:
        assert upper > lower

def test_empty_prediction_set():
    pass
