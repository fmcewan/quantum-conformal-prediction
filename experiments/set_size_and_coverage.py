import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcp.prediction.conformal_predictor import ConformalPredictor
from qcp.distributions.distribution_manager import create_distribution, event_probability
from qcp.utilities.file_handling import load_yaml

ALGORITHM_STYLES = {
    'dis': {'color': 'blue', 'linestyle': '-'},
    'knn': {'color': 'orange', 'linestyle': '--'},
    'mnn': {'color': 'green', 'linestyle': '-.'},
    '1nn': {'color': 'purple', 'linestyle': ':'},
}

def run(protocol_path, figure_id):

    configuration = load_yaml(protocol_path)
    common = configuration['common_properties']

    model_config = load_yaml(f"./data/models/{common['model_name']}/config.yaml")
    true_dist = create_distribution(model_config['data'])

    repeats = common['repeats']
    n_validation_points = common['validation_data_size']
    alphas_to_test = [common['alpha']]

    all_results = []

    for alpha in alphas_to_test:
        print(f"--- Running for alpha = {alpha:.2f} ---")
        for algorithm in configuration['algorithm_profiles']:
            current_configuration = (common | algorithm).copy()
            current_configuration['alpha'] = alpha

            cp = ConformalPredictor(current_configuration)

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

                total_set_size += set_size_sum / n_validation_points
                total_coverage += coverage_sum / n_validation_points

            all_results.append({
                'alpha': alpha,
                'algorithm': algorithm['name'],
                'avg_set_size': total_set_size / repeats,
                'avg_coverage': total_coverage / repeats
            })

    output_dir = f"data/figures/figure_{figure_id}"
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/results.csv", index=False)
    print(f"Results saved to {output_dir}/results.csv")

    return results_df

def plot(results_df, figure_id):

    output_dir = f"data/figures/figure_{figure_id}"
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Significance Level (α)')
    ax1.set_ylabel('Coverage', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='-', linewidth=1)
    ax1.plot(
        results_df['alpha'].unique(),
        1 - results_df['alpha'].unique(),
        'r-', label='Target Coverage (1-α)'
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Set Size', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    for name, group in results_df.groupby('algorithm'):
        style = ALGORITHM_STYLES.get(name, {'color': 'black', 'linestyle': '-'})
        ax1.plot(group['alpha'], group['avg_coverage'],
                 label=f'{name} Coverage',
                 color=style['color'],
                 linestyle=style['linestyle'])
        ax2.plot(group['alpha'], group['avg_set_size'],
                 label=f'{name} Set Size',
                 color=style['color'],
                 linestyle=style['linestyle'],
                 alpha=0.5)

    fig.suptitle('Coverage and Set Size vs. Significance Level', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.94, 0.861))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/figure_{figure_id}.pdf")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set size and coverage experiment")
    parser.add_argument("protocol", type=str, help="Path to protocol YAML file")
    parser.add_argument("figure_id", type=str, help="Figure output ID")
    args = parser.parse_args()

    results = run(args.protocol, args.figure_id)
    plot(results, args.figure_id)
