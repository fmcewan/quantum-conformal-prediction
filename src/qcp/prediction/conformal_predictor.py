# Standard library imports
import math

# Third-party imports
import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity

# Local imports
from qcp.prediction.scoring_functions import * 

from qcp.distributions.distribution_manager import create_distribution
from qcp.models.circuits.circuit_manager import CircuitManager 

from qcp.utilities.file_handling import load_yaml
from qcp.utilities.graphing_tricks import calculate_ranges

class ConformalPredictor:

    def __init__(self, predictor_configuration):
        
        print(f"Initialising conformal prediction algorithm: {predictor_configuration}")

        # Extract configuration variables
        self.calibration_data_size = predictor_configuration['calibration_data_size']
        self.alpha = predictor_configuration['alpha']
        self.score_function = predictor_configuration['score_function']
        self.M = predictor_configuration['M']
        self.model_name = predictor_configuration['model_name']

        # Initialize the model and distribution
        self.model = CircuitManger(self.model_name, predictor_configuration['hardware'])
        training_configuration = load_yaml(f"./saved/models/{self.model_name}/config.yaml")
        self.distribution = create_distribution(training_configuration['data'])

        # Initialize additional parameters and job data
        self.k = math.ceil(math.sqrt(self.M))
        self.start_idx = 0
        self.job_id_file_path = f"saved/jobs/{self.model.hardware}_{self.model_name}_M{self.M}.csv"
        self.jobs_df = pd.read_csv(self.job_id_file_path)

    def draw_from_jobs_df(self, chunk_size):
        
        if self.start_idx + chunk_size > len(self.jobs_df):
            raise ValueError("Requested chunk exceeds DataFrame length.")
        chunk = self.jobs_df.iloc[self.start_idx : self.start_idx + chunk_size]
        self.start_idx += chunk_size
        return chunk
    
    def calibrate(self):

        calibration_data = self.draw_from_jobs_df(self.calibration_data_size)
        self.scores = [
            self.score(y, job_id)
            for y, job_id in zip(calibration_data['y'], calibration_data['job_id'])
        ]
        q_level = np.ceil((self.calibration_data_size + 1) * (1 - self.alpha)) / self.calibration_data_size
        self.threshold = np.quantile(self.scores, q_level, method='higher')
        return self.threshold

    def generate_prediction_set_naive(self, job_id, kernel_function="gaussian", bandwidth=0.1, n_samples=100):

        self.model.extract_shots(job_id, self.M)
        model_samples_arr = []
        for value, frequency in self.model.data.items():
            model_samples_arr.extend([value] * frequency)
        model_samples_arr2d = np.array(model_samples_arr).reshape(-1, 1)

        # Fit the Kernel Density Estimation (KDE)
        kde = KernelDensity(kernel=kernel_function, bandwidth=bandwidth)
        kde.fit(model_samples_arr2d)
        
        # Create a grid and calculate the log-density
        grid = np.linspace(self.model.y_range[0], self.model.y_range[1], n_samples).reshape(-1, 1)
        log_density = kde.score_samples(grid)
        density = np.exp(log_density)

        # Sort the grid points by density in descending order
        sorted_indices = np.argsort(-density)
        sorted_density = density[sorted_indices]
        sorted_grid = grid[sorted_indices]

        # Calculate the cumulative sum and normalize the density
        cumulative_density = np.cumsum(sorted_density)
        cumulative_density /= cumulative_density[-1]

        # Find the cutoff index corresponding to the top 1-alpha quantile
        cutoff_index = np.searchsorted(cumulative_density, 1 - self.alpha)

        # Construct and sort the prediction set
        prediction_set = sorted_grid[:cutoff_index + 1]
        prediction_set_sorted = np.sort(np.array(prediction_set).flatten())
        ranges = calculate_ranges(prediction_set_sorted, self.model.y_range[0], self.model.y_range[1], n_samples)

        return ranges

    
    def generate_prediction_set(self, n_samples=100, job_id_for_prediction=None):
        """
        Generate a prediction set based on the specified scoring function.

        Depending on the scoring function defined in the configuration,
        this method either uses a naive KDE-based approach or computes scores for a grid of y-values,
        selecting those below a computed threshold.

        Parameters:
            n_samples (int, optional): Number of grid samples. Defaults to 100.

        Returns:
            list: A list of 2 integer lists representing prediction set in the form of a list of closed intervals.
        """

        if job_id_for_prediction:
            job_id = job_id_for_prediction
        else:
            test_data = self.draw_from_jobs_df(1)
            job_id = test_data['job_id'].values[0]

        if self.score_function == "naive":
            return self.generate_prediction_set_naive(job_id, n_samples=n_samples)
        
        test_y_points = np.linspace(self.model.y_range[0], self.model.y_range[1], n_samples)
        test_scores = [self.score(y, job_id) for y in test_y_points]
        prediction_points = test_y_points[test_scores <= self.threshold]
        prediction_set = calculate_ranges(prediction_points, self.model.y_range[0], self.model.y_range[1], n_samples)
        
        return prediction_set

    def score(self, y, job_id):

        self.model.extract_shots(job_id, self.M)
        
        if self.score_function == "dis":
            return scoring_functions.euclidean_distance(y, self.model.data)
        elif self.score_function == "1nn":
            return scoring_functions.nearest_neighbours(y, self.model.data, 1)
        elif self.score_function == "knn":
            return scoring_functions.nearest_neighbours(y, self.model.data, self.k)
        elif self.score_function == "mnn":
            return scoring_functions.nearest_neighbours(y, self.model.data, self.M)
        elif self.score_function == "hist":
            return scoring_functions.histogram(y, self.model.data, self.M, self.tau)
        else:
            raise NotImplementedError
