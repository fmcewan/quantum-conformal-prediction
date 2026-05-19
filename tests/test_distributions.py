import pytest
import numpy as np

from qcp.distributions.combined_normals import CombinedNormals
from qcp.distributions.normal import Normal
from qcp.distributions.distribution_manager import create_distribution, event_probability

# CombinedNormals tests
@pytest.fixture
def single_normal():
    return CombinedNormals([0], [1])

@pytest.fixture
def two_component():
    return CombinedNormals([-1, 1], [0.5, 0.5])

def test_combined_normals_invalid_init():
    with pytest.raises(ValueError):
        CombinedNormals([0, 1], [1])

def test_combined_normals_empty_init():
    with pytest.raises(ValueError):
        CombinedNormals([], [])

def test_pdf_integrates_to_one(single_normal):
    x = np.linspace(-5, 5, 10000)
    integral = np.trapezoid(single_normal._pdf(x), x)
    
    assert integral == pytest.approx(1.0, abs=1e-3)

def test_pdf_integrates_to_one_two_component(two_component):
    x = np.linspace(-5, 5, 10000)
    integral = np.trapezoid(two_component._pdf(x), x)
    
    assert integral == pytest.approx(1.0, abs=1e-3)

def test_pdf_non_negative(single_normal):
    x = np.linspace(-5, 5, 1000)
    
    assert np.all(single_normal._pdf(x) >= 0)

def test_cdf_monotonically_increasing(single_normal):
    x = np.linspace(-5, 5, 100)
    cdf_vals = [single_normal.cdf(xi) for xi in x]
    
    assert all(cdf_vals[i] <= cdf_vals[i+1] for i in range(len(cdf_vals) - 1))

def test_cdf_approaches_zero(single_normal):
    assert single_normal.cdf(-10) == pytest.approx(0.0, abs=1e-5)

def test_cdf_approaches_one(single_normal):
    assert single_normal.cdf(10) == pytest.approx(1.0, abs=1e-5)

def test_rvs_correct_size(single_normal):
    samples = single_normal.rvs(size=100)
    
    assert len(samples) == 100

def test_rvs_correct_size_two_component(two_component):
    samples = two_component.rvs(size=50)
    
    assert len(samples) == 50

# Normal tests
@pytest.fixture
def standard_normal():
    return Normal(loc=0, scale=1)

def test_normal_cdf_at_mean(standard_normal):
    assert standard_normal.cdf(0) == pytest.approx(0.5)

def test_normal_cdf_approaches_zero(standard_normal):
    assert standard_normal.cdf(-10) == pytest.approx(0.0, abs=1e-5)

def test_normal_cdf_approaches_one(standard_normal):
    assert standard_normal.cdf(10) == pytest.approx(1.0, abs=1e-5)

def test_normal_get_hdr(standard_normal):
    hdr = standard_normal.get_hdr(0.95)
    
    assert len(hdr) == 1
    
    lower, upper = hdr[0]
    
    assert lower < 0 < upper

def test_normal_hdr_width_increases_with_mass(standard_normal):
    
    hdr_90 = standard_normal.get_hdr(0.90)[0]
    hdr_95 = standard_normal.get_hdr(0.95)[0]
    width_90 = hdr_95[1] - hdr_90[0]
    width_95 = hdr_95[1] - hdr_95[0]
    
    assert width_95 > width_90

# create_distribution tests

def test_create_distribution_combined_normals():
    
    configuration = {
        "distribution": "combined_normals",
        "component_means": [0],
        "component_stds": [1]
    }
    distribution = create_distribution(configuration)
    
    assert isinstance(distribution, CombinedNormals)

def test_create_distribution_normal():
    
    configuration = {
        "distribution": "normal",
        "mean": 0,
        "scale": 1
    }
    distribution = create_distribution(configuration)
    
    assert isinstance(distribution, Normal)

def test_create_distribution_unknown_raises():
    
    configuration = {"distribution": "unknown"}
    with pytest.raises(ValueError):
        create_distribution(configuration)

# event_probability tests
def test_event_probability_full_range(standard_normal):
    event = [(-10, 10)]
    prob = event_probability(standard_normal, event)
    
    assert prob == pytest.approx(1.0, abs=1e-4)

def test_event_probability_empty_event(standard_normal):
    prob = event_probability(standard_normal, [])
    
    assert prob == pytest.approx(0.0)

def test_event_probability_in_range(standard_normal):
    event = [(-1, 1)]
    prob = event_probability(standard_normal, event)
    
    assert 0 < prob < 1

def test_event_probability_two_intervals(two_component):
    event = [(-2, -0.5), (0.5, 2)]
    prob = event_probability(two_component, event)

    assert 0 < prob < 1
