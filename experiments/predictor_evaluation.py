import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcp.distributions.distribution_manager import create_distribution, event_probability
from qcp.utilities.file_handling import load_yaml

ALGORITHM_STYLES = {
    'dis': {'color': 'blue', 'linestyle': '-'},
    'knn': {'color': 'orange', 'linestyle': '--'},
    'mnn': {'color': 'green', 'linestyle': '-.'},
    '1nn': {'color': 'purple', 'linestyle': ':'},
}


def run(protocol_name):
    protocol = load_yaml(f'protocols/{protocol_name}.yml')
    excluded_keys = {'name', 'algorithms'}
    common = {k: v for k, v in protocol.items() if k not in excluded_keys}

    model_config = load_yaml(f"./data/models/{common['model_name']}/configuration.yml")
    true_dist = create_distribution(model_config['data'])

    results_dir = f'data/results/{protocol_name}'
    figures_dir = f'data/figures/{protocol_name}'
    os.makedirs(figures_dir, exist_ok=True)

    all_results = []

    for algorithm in protocol['algorithms']:
        name = algorithm['name']
        df = pd.read_csv(f'{results_dir}/results_{name}.csv')
        df['prediction_set'] = df['prediction_set'].apply(json.loads)

        set_sizes = [
            sum(upper - lower for lower, upper in row)
            for row in df['prediction_set']
        ]
        coverages = [
            event_probability(true_dist, row)
            for row in df['prediction_set']
        ]

        all_results.append({
            'algorithm': name,
            'alpha': df['alpha'].iloc[0],
            'avg_set_size': np.mean(set_sizes),
            'avg_coverage': np.mean(coverages)
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{figures_dir}/results.csv', index=False)
    return results_df


def plot(results_df, protocol_name):
    figures_dir = f'data/figures/{protocol_name}'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Coverage', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='-', linewidth=1)
    ax1.axhline(
        y=1 - results_df['alpha'].iloc[0],
        color='r', linestyle='-',
        label=f"Target Coverage (1-α={1 - results_df['alpha'].iloc[0]:.2f})"
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Set Size', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    for _, row in results_df.iterrows():
        style = ALGORITHM_STYLES.get(row['algorithm'], {'color': 'black', 'linestyle': '-'})
        ax1.bar(row['algorithm'], row['avg_coverage'], color=style['color'], alpha=0.6,
                label=f"{row['algorithm']} Coverage")
        ax2.plot(row['algorithm'], row['avg_set_size'], marker='o', color=style['color'],
                 label=f"{row['algorithm']} Set Size")

    fig.suptitle('Coverage and Set Size by Algorithm', fontsize=16)
    fig.legend(loc='upper right', bbox_to_anchor=(0.94, 0.861))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{figures_dir}/predictor_evaluation.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictor evaluation experiment')
    parser.add_argument('protocol', type=str, help='Protocol name (e.g. predictor_evaluation)')
    args = parser.parse_args()

    results = run(args.protocol)
    plot(results, args.protocol)
