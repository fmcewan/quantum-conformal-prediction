import math
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from qcp.prediction.scoring_functions import (
    euclidean_distance,
    nearest_neighbours,
    histogram
)

from qcp.distributions.distribution_manager import create_distribution
from qcp.models.circuits.circuit_manager import CircuitManager
from qcp.utilities.file_handling import load_yaml
from qcp.utilities.graphing_tricks import calculate_ranges

class ConformalPredictor:

    def __init__(self, predictor_configuration):
        print(f"Initialising conformal predictor: {predictor_configuration}")

        self.calibration_data_size = predictor_configuration['calibration_data_size']
        self.alpha = predictor_configuration['alpha']
        self.score_function = predictor_configuration['score_function']
        self.M = predictor_configuration['M']
        self.model_name = predictor_configuration['model_name']

        self.model = CircuitManager(self.model_name, predictor_configuration['hardware'])
        training_configuration = load_yaml(f"./data/models/{self.model_name}/config.yaml")
        self.distribution = create_distribution(training_configuration['data'])

        self.k = math.ceil(math.sqrt(self.M))
        self.job_id_file_path = f"data/jobs/{self.model.hardware}_{self.model_name}_M{self.M}.csv"
        self.jobs_df = pd.read_csv(self.job_id_file_path)

    def sample_jobs(self, n, replace=False):

        if n > len(self.jobs_df):
            raise ValueError(f"Requested {n} jobs but only {len(self.jobs_df)} available.")
        return self.jobs_df.sample(n=n, replace=replace)

    def calibrate(self):

        calibration_data = self.sample_jobs(self.calibration_data_size)
        
        self.scores = []
        for y, job_id in zip(calibration_data['y'], calibration_data['job_id']):
            self.model.extract_shots(job_id, self.M)
            self.scores.append(self.score(y))
        
        q_level = np.ceil((self.calibration_data_size + 1) * (1 - self.alpha)) / self.calibration_data_size
        self.threshold = np.quantile(self.scores, q_level, method='higher')
        
        return self.threshold    

    def generate_prediction_set(self, n_samples=100, job_id=None):

        if job_id is None:
            test_data = self.sample_jobs(1)
            job_id = test_data['job_id'].values[0]

        self.model.extract_shots(job_id, self.M)

        if self.score_function == "naive":
            
            model_samples = []
            for value, frequency in self.model.data.items():
                model_samples.extend([value] * frequency)
            
            kde = KernelDensity(kernel="gaussian", bandwidth=0.1)
            kde.fit(np.array(model_samples).reshape(-1, 1))
            
            grid = np.linspace(self.model.y_range[0], self.model.y_range[1], n_samples).reshape(-1, 1)
            density = np.exp(kde.score_samples(grid))
            
            sorted_indices = np.argsort(-density)
            cumulative_density = np.cumsum(density[sorted_indices])
            cumulative_density /= cumulative_density[-1]
            
            cutoff_index = np.searchsorted(cumulative_density, 1 - self.alpha)
            prediction_set = np.sort(grid[sorted_indices[:cutoff_index + 1]].flatten())
        
        else:
            
            test_y_points = np.linspace(self.model.y_range[0], self.model.y_range[1], n_samples)
            test_scores = np.array([self.score(y) for y in test_y_points])
            prediction_set = test_y_points[test_scores <= self.threshold]

        return calculate_ranges(prediction_set, self.model.y_range[0], self.model.y_range[1], n_samples)

    def score(self, y):
    
        match self.score_function:
            case "dis":
                return euclidean_distance(y, self.model.data)
            case "1nn":
                return nearest_neighbours(y, self.model.data, 1)
            case "knn":
                return nearest_neighbours(y, self.model.data, self.k)
            case "mnn":
                return nearest_neighbours(y, self.model.data, self.M)
            case "hist":
                return histogram(y, self.model.data, self.M, self.tau)
            case _:
                raise NotImplementedError(f"Score function '{self.score_function}' is not implemented.")
