import pytest

from qcp.prediction.scoring_functions import euclidean_distance, nearest_neighbours

# euclidean_distance tests
def test_euclidean_distance_exact_match():
    result = euclidean_distance(1.0, {1.0: 10})
    
    assert result == pytest.approx(0.0)

def test_euclidean_distance_single_value():
    result = euclidean_distance(0.0, {2.0: 1})

    assert result == pytest.approx(2.0)

def test_euclidean_distance_weighted_mean():
    result = euclidean_distance(0.0, {0.0: 1, 4.0: 1})
    
    assert result == pytest.approx(2.0)

def test_euclidean_distance_non_negative():
    result = euclidean_distance(5.0, {1.0: 3, 2.0: 7})
    
    assert result >= 0

def test_euclidean_distance_symmetric():
    freq_dict = {0.0: 1, 4.0: 1} 
    
    assert euclidean_distance(0.0, freq_dict) == pytest.approx(euclidean_distance(4.0, freq_dict))

# nearest_neighbours tests
def test_nearest_neighbours_k1():
    result = nearest_neighbours(0.0, {1.0: 1, 3.0: 1, 5.0: 1}, k=1)

    assert result == pytest.approx(1.0)

def test_nearest_neighbours_k2():
    result = nearest_neighbours(0.0, {1.0: 1, 3.0: 1, 5.0: 1}, k=2)

    assert result == pytest.approx(3.0)

def test_nearest_neighbours_k3():
    result = nearest_neighbours(0.0, {1.0: 1, 3.0: 1, 5.0: 1}, k=3)

    assert result == pytest.approx(5.0)

def test_nearest_neighbours_respects_frequency():
    result = nearest_neighbours(0.0, {1.0: 3, 5.0: 1}, k=3)

    assert result == pytest.approx(1.0)

def test_nearest_neighbours_exact_match():
    result = nearest_neighbours(2.0, {2.0: 1, 4.0: 1}, k=1)

    assert result == pytest.approx(0.0)

def test_nearest_neighbours_non_negative():
    result = nearest_neighbours(3.0, {1.0: 2, 5.0: 2}, k=1)

    assert result >= 0

# TODO: Test histogram scoring function
