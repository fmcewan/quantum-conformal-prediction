import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qcp.utilities.file_handling import load_yaml


def run(protocol_names):
    all_results = []

    for protocol_name in protocol_names:
        protocol = load_yaml(f'protocols/{protocol_name}.yml')
        excluded_keys = {'name', 'algorithms'}
        common = {k: v for k, v in protocol.items() if k not in excluded_keys}

        model_config = load_yaml(f"./data/models/{common['model_name']}/configuration.yml")
        model_type = model_config['model']['type']

        results_dir = f'data/results/{protocol_name}'

        for algorithm in protocol['algorithms']:
            name = algorithm['name']
            df = pd.read_csv(f'{results_dir}/results_{name}.csv')
            df['prediction_set'] = df['prediction_set'].apply(json.loads)

            if model_type == 'classification':
                set_sizes = [len(row) for row in df['prediction_set']]
            else:
                set_sizes = [
                    sum(upper - lower for lower, upper in row)
                    for row in df['prediction_set']
                ]

            all_results.append({
                'label': f"{protocol_name} / {name}",
                'model_type': model_type,
                'avg_set_size': np.mean(set_sizes),
                'set_size_std': np.std(set_sizes)
            })

    results_df = pd.DataFrame(all_results)

    output_dir = f'data/figures/informativeness'
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f'{output_dir}/results.csv', index=False)

    return results_df


def plot(results_df):
    figures_dir = 'data/figures/informativeness'

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']

    bars = ax.bar(
        results_df['label'],
        results_df['avg_set_size'],
        yerr=results_df['set_size_std'],
        capsize=5,
        color=colors[:len(results_df)],
        alpha=0.8
    )

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.05,
            f'{yval:.2f}',
            ha='center',
            va='bottom',
            fontsize=11
        )

    ax.set_ylabel('Average Prediction Set Size', fontsize=12)
    ax.set_title('Informativeness', fontsize=14, pad=20)
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'{figures_dir}/informativeness.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Informativeness experiment')
    parser.add_argument('protocols', nargs='+', type=str,
                        help='One or more protocol names to compare')
    args = parser.parse_args()

    results = run(args.protocols)
    plot(results)
