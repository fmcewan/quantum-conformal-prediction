import pytest
import torch
import numpy as np

from qcp.utilities.metrics import NegativeLogSumCriterion
from qcp.distributions.skewed_normal import SkewedNormal
from qcp.distributions.heteroscedastic import Heteroscedastic
from qcp.distributions.sinusoidal import Sinusoidal
from qcp.distributions.random_gibbs_states import RandomGibbsStates


# NegativeLogSumCriterion tests
@pytest.fixture
def criterion():
    return NegativeLogSumCriterion()

def test_criterion_high_probability_low_loss(criterion):
    high_probabilities = torch.tensor([0.9, 0.8, 0.95])
    low_probabilities = torch.tensor([0.1, 0.2, 0.05])

    assert criterion(high_probabilities) < criterion(low_probabilities)

def test_criterion_clamps_near_zero(criterion):
    probabilities = torch.tensor([0.0, 0.5, 1.0])
    result = criterion(probabilities)
    assert torch.isfinite(result)

def test_criterion_returns_scalar(criterion):
    probabilities = torch.tensor([0.5, 0.3, 0.2])
    result = criterion(probabilities)
    assert result.shape == torch.Size([])

def test_criterion_negative_log(criterion):
    probabilities = torch.tensor([1.0, 1.0, 1.0])
    result = criterion(probabilities)
    assert result.item() == pytest.approx(0.0, abs=1e-5)


# SkewedNormal tests
@pytest.fixture
def skewed_distribution():
    return SkewedNormal(loc=0, scale=1, skew=2)

def test_skewed_normal_pdf_non_negative(skewed_distribution):
    x = np.linspace(-5, 5, 1000)
    assert np.all(skewed_distribution._pdf(x) >= 0)

def test_skewed_normal_pdf_integrates_to_one(skewed_distribution):
    x = np.linspace(-5, 5, 10000)
    integral = np.trapezoid(skewed_distribution._pdf(x), x)
    assert integral == pytest.approx(1.0, abs=1e-2)

def test_skewed_normal_cdf_approaches_zero(skewed_distribution):
    assert skewed_distribution.cdf(-10) == pytest.approx(0.0, abs=1e-4)

def test_skewed_normal_cdf_approaches_one(skewed_distribution):
    assert skewed_distribution.cdf(10) == pytest.approx(1.0, abs=1e-4)

def test_skewed_normal_rvs_correct_size(skewed_distribution):
    samples = skewed_distribution.rvs(size=100)
    assert len(samples) == 100


# Heteroscedastic tests

@pytest.fixture
def heteroscedastic_distribution():
    return Heteroscedastic(x_range=[1, 5])

def test_heteroscedastic_rvs_correct_size(heteroscedastic_distribution):
    x_points, y_points = heteroscedastic_distribution.rvs(size=50)
    assert len(x_points) == 50
    assert len(y_points) == 50

def test_heteroscedastic_x_in_range(heteroscedastic_distribution):
    x_points, _ = heteroscedastic_distribution.rvs(size=100)
    assert np.all(x_points >= 1)
    assert np.all(x_points <= 5)

def test_heteroscedastic_std_increases_with_x(heteroscedastic_distribution):
    assert heteroscedastic_distribution.component_std(1) < heteroscedastic_distribution.component_std(5)

def test_heteroscedastic_cdf_given_x(heteroscedastic_distribution):
    prob = heteroscedastic_distribution.cdf_given_x(100, 3.0)
    assert prob == pytest.approx(1.0, abs=1e-4)

# Sinusoidal tests
@pytest.fixture
def sinusoidal_distribution():
    return Sinusoidal(x_range=[-5, 5])

def test_sinusoidal_rvs_correct_size(sinusoidal_distribution):
    x_points, y_points = sinusoidal_distribution.rvs(size=50)
    assert len(x_points) == 50
    assert len(y_points) == 50

def test_sinusoidal_x_in_range(sinusoidal_distribution):
    x_points, _ = sinusoidal_distribution.rvs(size=100)
    assert np.all(x_points >= -5)
    assert np.all(x_points <= 5)

def test_sinusoidal_component_mean_is_scalar(sinusoidal_distribution):
    result = sinusoidal_distribution.component_mean(1.0)
    assert np.isscalar(result)

def test_sinusoidal_component_std_is_constant(sinusoidal_distribution):
    assert sinusoidal_distribution.component_std() == pytest.approx(0.05)


# RandomGibbsStates tests

@pytest.fixture
def gibbs_distribution():
    return RandomGibbsStates(
        num_classes=4,
        num_features=10,
        dimension=2,
        density=0.5,
        temperature=1.0
    )

def test_gibbs_generate_data_correct_size(gibbs_distribution):
    features, labels = gibbs_distribution.generate_data()
    assert len(features) == 10
    assert len(labels) == 10

def test_gibbs_labels_in_range(gibbs_distribution):
    _, labels = gibbs_distribution.generate_data()
    assert torch.all(labels >= 0)
    assert torch.all(labels < gibbs_distribution.num_classes)

def test_gibbs_state_is_valid_density_matrix(gibbs_distribution):
    import scipy
    hamiltonian = scipy.sparse.random(2, 2, density=0.5, format='csr', dtype=np.complex128).toarray()
    rho = gibbs_distribution.generate_gibbs_state(hamiltonian)
    assert np.trace(rho) == pytest.approx(1.0, abs=1e-5)
    assert rho.shape == (2, 2)

def test_gibbs_class_label_in_range(gibbs_distribution):
    import scipy
    hamiltonian = scipy.sparse.random(2, 2, density=0.5, format='csr', dtype=np.complex128).toarray()
    rho = gibbs_distribution.generate_gibbs_state(hamiltonian)
    label = gibbs_distribution.generate_class_label_from_gibbs_state(rho)
    assert 0 <= label < gibbs_distribution.num_classes
