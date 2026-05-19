# Default imports
import numpy as np
import pandas as pd

# File imports
from qcp.prediction.conformal_predictor import ConformalPredictor 
from qcp.models.circuits.circuit_manager import CircuitManager
from qcp.utilities.file_handling import load_yaml
from qcp.distributions.distribution_manager import create_distribution, event_probability

def measurement_histograms(results_config_path, figure_id):
    """
    Generate and save from a range of models
    
    Parameters:
        results_config_path (str): Path to the configuration file for results.
        figure_id (int): Identifier for the output figure folder.
    """

    # Load configuration for results
    results_config = load_yaml(f"configurations/results/{results_config_path}")
    model = CircuitManager(results_config['common_properties']['model_name'], "aer")
    M = results_config['common_properties']['M']
    
    # Create column names for the DataFrame
    column_names = ['eigenstates', 'y_values'] + [algo_config['name'] for algo_config in results_config['algorithm_profiles']]
    df = pd.DataFrame(columns=column_names)

    # Ensure column names are unique
    if not df.columns.is_unique:
        raise ValueError("Column names are not unique!")

    num_states = 2**model.num_qubits
    df['y_values'] = np.linspace(model.y_range[0], model.y_range[1], num_states)
    df['eigenstates'] = [f"{bin(i)[2:].zfill(model.num_qubits)}" for i in range(num_states)]

    # Process each algorithm configuration
    for algo_config in results_config['algorithm_profiles']:
        merged_config = results_config.get('common_properties') | algo_config 
        model.set_hardware(merged_config['hardware'])
        job_id_file_path = f"data/jobs/{merged_config['hardware']}_{merged_config['model_name']}_M{merged_config['M']}.csv"
        data_df = pd.read_csv(job_id_file_path)
        example_job_id = str(data_df.iloc[0]['job_id'])
        model.extract_shots(example_job_id, M)

        # Normalize and map data to the DataFrame
        normalized_data = {key: value / M for key, value in model.data_binary.items()}
        df[merged_config['name']] = df['eigenstates'].map(normalized_data).fillna(0).astype(float)

    # Save results and config file
    output_path = f"data/figures/figure_{figure_id}/results.csv"
    df.to_csv(output_path, index=False)
    save_config_in_figure_folder(figure_id, results_config_path)

def unsupervised_prediction_sets(results_config_path, figure_id):
    """
    Generates prediction sets using conformal algorithms on unsupervised models.

    This method reads a configuration file from the specified `results_config_path`. The configuration
    file includes a base algorithm with parameters such as calibration_data_size, M, job_id_file_name, and alpha,
    along with a list of algorithms. Each algorithm entry contains a name, model_name, hardware, and a scoring function.
    The method processes these configurations to generate prediction sets and computes coverage metrics for each algorithm.
    The updated configuration, now augmented with additional fields for coverage and prediction sets, is data to the folder
    specified by `figure_id`.

    Args:
        results_config_path (str): Path to the configuration file with algorithm specifications.
        figure_id (str): Identifier for the folder where the updated configuration file with prediction sets will be data.

    Returns:
        None
    """

    # Load configurations
    results_config = load_yaml(f"configurations/results/{results_config_path}")

    model_config_path = f"./data/models/{results_config['common_properties']['model_name']}/config.yaml"
    model_config = load_yaml(model_config_path)
    
    true_dist = create_distribution(model_config['data'])

    # Process each algorithm
    for algo_config in results_config['algorithm_profiles']:
        # Update algorithm-specific configuration
        merged_config = results_config.get('common_properties') | algo_config 

        # Initialize Conformal Prediction algorithm
        cp = ConformalPredictor(merged_config)
        cp.calibrate()

        # Generate prediction set
        print(f"\nProcessing algorithm: {algo_config['name']}")
        print("Generating prediction set...")
        prediction_set = cp.generate_prediction_set()
        prediction_set = [[x.item() for x in interval] for interval in prediction_set]

        # Calculate coverage
        print("Calculating coverage...")
        coverage = event_probability(true_dist, prediction_set)
        print(f"Coverage: {coverage}%\n")

        # Store results for the algorithm
        algo_config['prediction_set'] = prediction_set
        algo_config['coverage'] = coverage.item()

    # Save results
    save_config_in_figure_folder(figure_id, results_config, config_name="results.yaml")

# Unsupervised experiments

def set_size_and_coverage(results_configuration_path, figure_id):

    # Load configurations
    results_configuration = load_yaml(f"configurations/results/{results_configuration_path}")
    
    model_configuration_path = f"./data/models/{results_configuration['common_properties']['model_name']}/config.yaml"
    model_configuration = load_yaml(model_configuration_path)
    
    # Create distribution and parameters
    true_dist = create_distribution(model_configuration['data'])
    repeats = results_configuration['common_properties']['repeats']
    M = results_configuration['common_properties']['M']
    n_validation_points = results_configuration['common_properties']['validation_data_size']
    
    # Generating different epsilon values
    alphas_to_test = [0.05]
    all_results = []

    # Initialize DataFrame
    for alpha in alphas_to_test:
        print(f"--- Running for alpha = {alpha:.2f} ---")
        for algorithm_configuration in results_configuration['algorithm_profiles']:
        # Merge base algorithm settings with specific algorithm settings
            current_configuration = (results_configuration['common_properties'] | algorithm_configuration).copy()
            current_configuration['alpha'] = alpha

            cp = ConformalPredictor(current_configuration)

            # Initialize arrays for metrics
            total_set_size = 0
            total_coverage = 0

            for r in range(repeats):
                print(f"Repeat = {r+1}/{repeats}")
                set_size_sum = 0
                coverage_sum = 0

                cp.calibrate()
                for _ in range(n_validation_points):
                    prediction_set = cp.generate_prediction_set()
                    set_size_sum += sum(interval[1] - interval[0] for interval in prediction_set)
                    coverage_sum += event_probability(true_dist, prediction_set)

                # Compute averages for this value of m
                total_set_size += set_size_sum / n_validation_points
                total_coverage += coverage_sum / n_validation_points
            
            final_set_size_avg = total_set_size / repeats
            final_coverage_avg = total_coverage / repeats

            all_results.append({
                    'alpha': alpha,
                    'algorithm': algorithm_configuration['name'],
                    'avg_set_size': final_set_size_avg,
                    'avg_coverage': final_coverage_avg
                })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"data/figures/figure_{figure_id}/results.csv", index=False)
    
    save_config_in_figure_folder(figure_id, results_configuration_path)

def unsupervised_informativeness(results_configuration_path, figure_id):
    """
    Generates summary data for the informativeness figure by calculating the
    mean and standard deviation of set sizes over multiple repeats.

    Args:
        results_configuration_path (str): Path to the results config YAML file.
        figure_id (str): Identifier for the folder where the output CSV will be data.
    """
    # Load configurations
    results_configuration = load_yaml(f"configurations/results/{results_configuration_path}")
    
    # Get the separate model configurations
    model_configuration_names = []
    for algorithm_configuration in results_configuration['algorithm_profiles']:
        model_configuration_name = algorithm_configuration['model_name']
        model_configuration_names.append(model_configuration_name)
        
        algorithm_configuration.pop('model_name', None)
   
    # Retrieve the model configuration to ascertain the data that needs to be generate_prediction_set

    model_configuration_path = f"./data/models/{model_configuration_names[0]}/config.yaml"
    model_configuration = load_yaml(model_configuration_path)
    
    # Retrieve the true distribution
    true_distribution = create_distribution(model_configuration['data'])

    # Initialising required properties for data generation
    repeats = results_configuration['common_properties']['repeats']
    n_validation_points = results_configuration['common_properties']['validation_data_size']

    # Running the experiment for both models to generate the necessary data
    all_results = []
    
    count = 0
    for algorithm_configuration in results_configuration['algorithm_profiles']:
        
        # Retrieving current model configuration names
        model_configuration_name = model_configuration_names[count]
        count += 1

        print(f"--- Running for algorithm: {algorithm_configuration['name']} ---")
        
        print(f"--- Model name: " + model_configuration_name)
        current_configuration = (results_configuration['common_properties'] | algorithm_configuration).copy()
        current_configuration['model_name'] = model_configuration_name

        print(current_configuration['model_name'])

        cp = ConformalPredictor(current_configuration)

        repeat_avg_set_sizes = []
        # ----------------------------------------------------

        for r in range(repeats):
            print(f"Repeat = {r+1}/{repeats}")
            set_size_sum = 0

            cp.calibrate()
            for _ in range(n_validation_points):
                prediction_set = cp.generate_prediction_set()
                set_size_sum += sum(interval[1] - interval[0] for interval in prediction_set)

            avg_set_size_for_repeat = set_size_sum / n_validation_points
            repeat_avg_set_sizes.append(avg_set_size_for_repeat)
        
        final_avg = np.mean(repeat_avg_set_sizes)
        final_std = np.std(repeat_avg_set_sizes)

        print(f"Algorithm: {algorithm_configuration['name']}, Avg Set Size: {final_avg:.3f}, Std Dev: {final_std:.3f}")

        all_results.append({
            'model_quality': algorithm_configuration['name'], 
            'avg_set_size': final_avg,
            'set_size_std': final_std
        })

    results_df = pd.DataFrame(all_results)
    print(results_df.head())
    output_path = f"data/figures/figure_{figure_id}/results.csv"
    results_df.to_csv(output_path, index=False)
    
    save_config_in_figure_folder(figure_id, results_configuration_path)

# Supervised experiments

def prediction_intervals(results_configuration_path, figure_id):
    """
    Generates and saves the per-point prediction interval data for the
    supervised regression experiment.

    Args:
        results_configuration_path (str): Path to the results config YAML file.
        figure_id (str): Identifier for the folder where the output CSV will be data.
    """
    # Load configurations
    results_configuration = load_yaml(f"configurations/results/{results_configuration_path}")
    model_configuration_path = f"./data/models/{results_configuration['common_properties']['model_name']}/config.yaml"
    model_configuration = load_yaml(model_configuration_path)

    # --- This function only supports one algorithm profile at a time ---
    if len(results_configuration['algorithm_profiles']) > 1:
        raise ValueError("This function is designed to process only one algorithm profile at a time.")
    
    n_validation_points = results_configuration['common_properties']['validation_data_size']
    
    for algorithm_configuration in results_configuration['algorithm_profiles']:
        print(f"--- Running for algorithm: {algorithm_configuration['name']} ---")

        current_configuration = results_configuration['common_properties'] | algorithm_configuration

        cp = ConformalPredictor(current_configuration)

        print("--- Calibrating Model ---")
        cp.calibrate()
        print(f"Calibration complete. Threshold q = {cp.threshold:.4f}")

        all_results = []

        print(f"--- Generating Prediction Intervals for {n_validation_points} test points ---")
        for i in range(n_validation_points):
            if (i + 1) % 20 == 0:
                print(f"Processing point {i+1}/{n_validation_points}...")
                
            # Drawing a single test point
            test_data_point = cp.draw_from_jobs_df(1)
            x_true = test_data_point['x'].values[0]
            y_true = test_data_point['y'].values[0]
            job_id = test_data_point['job_id'].values[0]
            
            # Generating a single prediction interval             
            prediction_interval = cp.generate_prediction_set(job_id_for_prediction=job_id)
            
            # Getting the M shots for the current test point
            model_samples = []
            for value, frequency in cp.model.data.items():
                model_samples.extend([value] * frequency)
            y_hat = np.median(model_samples)
        
            # Assuming for regression, the prediction_set is a single interval [[lower, upper]]
            if prediction_interval and len(prediction_interval) > 0:
                y_lower = prediction_interval[0][0]
                y_upper = prediction_interval[0][1]
            else:
                # Handle cases where the prediction set is empty
                y_lower, y_upper = None, None

            all_results.append({
                'x': x_true,
                'y_true': y_true,
                'y_hat': y_hat,
                'y_lower': y_lower,
                'y_upper': y_upper
            })

    # Create and save the final DataFrame
    results_df = pd.DataFrame(all_results)
    output_path = f"data/figures/figure_{figure_id}/results.csv"
    results_df.to_csv(output_path, index=False)

def supervised_informativeness(results_configuration_path, figure_id):
    """
    Generates summary data for the supervised informativeness figure.
    This function should be run once for each model configuration 
    (e.g., good_classifier, bad_classifier, good_regressor, bad_regressor).
    """
    # Load configurations
    results_configuration = load_yaml(f"configurations/results/{results_configuration_path}")
    
    # Common parameters
    common_props = results_configuration['common_properties']
    repeats = common_props['repeats']
    n_validation_points = common_props['validation_data_size']
    alpha = common_props['alpha'] # Use the single alpha for this experiment

    # Since this is for one model at a time, we take the first profile
    algo_config = results_configuration['algorithm_profiles'][0]
    current_configuration = (common_props | algo_config).copy()
    current_configuration['alpha'] = alpha

    cp = ConformalPredictor(current_configuration)
    
    # Store the average set size from each repeat to calculate std dev later
    repeat_avg_set_sizes = []

    print(f"--- Running for model: {common_props['model_name']}, Algorithm: {algo_config['name']} ---")
    for r in range(repeats):
        print(f"Repeat = {r+1}/{repeats}")
        set_size_sum = 0
        
        cp.calibrate()
        for _ in range(n_validation_points):
            prediction_set = cp.generate_prediction_set()
            
            # Logic for set size calculation
            if cp.model.type == 'classification':
                set_size_sum += len(prediction_set)
            else: # For regression
                set_size_sum += sum(interval[1] - interval[0] for interval in prediction_set)

        # Calculate the average set size for this one repeat
        avg_set_size_for_repeat = set_size_sum / n_validation_points
        repeat_avg_set_sizes.append(avg_set_size_for_repeat)

    # --- Calculate final mean and std dev over all repeats ---
    final_avg = np.mean(repeat_avg_set_sizes)
    final_std = np.std(repeat_avg_set_sizes)
    
    print(f"Final Avg Set Size: {final_avg:.3f}, Std Dev: {final_std:.3f}")

    # Create and save the result for this single run
    result_data = {
        'task': [cp.model.type],
        'model_quality': [common_props.get('quality', 'N/A')], # Add a 'quality' key to your YAML
        'avg_set_size': [final_avg],
        'set_size_std': [final_std]
    }
    results_df = pd.DataFrame(result_data)
    
    # Save with a unique name based on the config
    output_path = f"data/figures/figure_{figure_id}/results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Informativeness data data to {output_path}")
