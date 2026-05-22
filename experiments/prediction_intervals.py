import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcp.utilities.file_handling import load_yaml


def run(protocol_name):
    protocol = load_yaml(f'protocols/{protocol_name}.yml')
    excluded_keys = {'name', 'algorithms'}
    common = {k: v for k, v in protocol.items() if k not in excluded_keys}

    results_dir = f'data/results/{protocol_name}'
    figures_dir = f'data/figures/{protocol_name}'
    os.makedirs(figures_dir, exist_ok=True)

    all_results = []

    for algorithm in protocol['algorithms']:
        name = algorithm['name']
        df = pd.read_csv(f'{results_dir}/results_{name}.csv')
        df['prediction_set'] = df['prediction_set'].apply(json.loads)

        for _, row in df.iterrows():
            intervals = row['prediction_set']
            y_lower = intervals[0][0] if intervals else None
            y_upper = intervals[-1][1] if intervals else None

            model_samples = []
            all_results.append({
                'algorithm': name,
                'x': row.get('x', 0),
                'y_true': row['y_true'],
                'y_lower': y_lower,
                'y_upper': y_upper,
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'{figures_dir}/results.csv', index=False)
    return results_df


def plot(results_df, protocol_name):
    figures_dir = f'data/figures/{protocol_name}'

    for algorithm_name, group in results_df.groupby('algorithm'):
        group = group.sort_values('x').dropna()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(group['x'], group['y_true'], color='black', s=10, alpha=0.5,
                   label='True Data')
        ax.fill_between(
            group['x'],
            group['y_lower'],
            group['y_upper'],
            color='red', alpha=0.2,
            label='Prediction Interval'
        )

        ax.set_xlabel('Input (x)', fontsize=12)
        ax.set_ylabel('Output (y)', fontsize=12)
        ax.set_title(f'Prediction Intervals — {algorithm_name}', fontsize=14, pad=20)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f'{figures_dir}/prediction_intervals_{algorithm_name}.pdf')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction intervals experiment')
    parser.add_argument('protocol', type=str, help='Protocol name (e.g. prediction_intervals)')
    args = parser.parse_args()

    results = run(args.protocol)
    plot(results, args.protocol)
