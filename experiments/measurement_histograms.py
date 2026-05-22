import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from qcp.models.circuits.circuit_manager import CircuitManager
from qcp.distributions.distribution_manager import create_distribution
from qcp.utilities.file_handling import load_yaml

base_size = 9
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': base_size,
    'axes.titlesize': base_size + 2,
    'axes.labelsize': base_size,
    'xtick.labelsize': base_size - 2,
    'ytick.labelsize': base_size - 2,
    'legend.fontsize': base_size - 1,
})

inside_title_args = {
    'fontsize': 9,
    'va': 'top',
    'ha': 'center',
    'color': 'red',
    'bbox': {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.0}
}


def run(protocol_name):
    protocol = load_yaml(f'protocols/{protocol_name}.yml')
    excluded_keys = {'name', 'algorithms'}
    common = {k: v for k, v in protocol.items() if k not in excluded_keys}

    model_name = common['model_name']
    M = common['M']

    model = CircuitManager(model_name, 'aer')
    num_states = 2 ** model.n_qubits

    column_names = ['eigenstates', 'y_values'] + [a['name'] for a in protocol['algorithms']]
    df = pd.DataFrame(columns=column_names)
    df['y_values'] = np.linspace(model.y_range[0], model.y_range[1], num_states)
    df['eigenstates'] = [f"{bin(i)[2:].zfill(model.n_qubits)}" for i in range(num_states)]

    for algorithm in protocol['algorithms']:
        merged = common | algorithm
        model.set_hardware(merged['hardware'])
        jobs_df = pd.read_csv(f"data/jobs/{merged['hardware']}_{model_name}_M{M}.csv")
        example_job_id = str(jobs_df.iloc[0]['job_id'])
        model.extract_shots(example_job_id, M)

        normalised = {k: v / M for k, v in model.data_binary.items()}
        df[merged['name']] = df['eigenstates'].map(normalised).fillna(0).astype(float)

    figures_dir = f'data/figures/{protocol_name}'
    os.makedirs(figures_dir, exist_ok=True)
    df.to_csv(f'{figures_dir}/results.csv', index=False)
    return df, model_name


def plot(df, model_name, protocol_name):
    figures_dir = f'data/figures/{protocol_name}'

    model_config = load_yaml(f'./data/models/{model_name}/configuration.yml')
    true_dist = create_distribution(model_config['data'])

    x_points = np.linspace(model_config['data']['y_range'][0], model_config['data']['y_range'][1], 1000)
    y_points = [true_dist._pdf(x) for x in x_points]

    algorithm_columns = df.columns[2:]
    n_plots = len(algorithm_columns) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))

    axes[0].plot(x_points, y_points, color='black')
    axes[0].text(0.5, 0.95, "Ground Truth Density", transform=axes[0].transAxes, **inside_title_args)
    axes[0].set_xlabel('y')

    for idx, column in enumerate(algorithm_columns):
        ax = axes[idx + 1]
        ax.bar(df['y_values'], df[column], ec='none', alpha=0.5, width=0.12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.text(0.5, 0.95, f"PQC Histogram\n({column})", transform=ax.transAxes, **inside_title_args)
        ax.set_xlabel('y')

    for ax in axes:
        ax.set_xlim([model_config['data']['y_range'][0] - 0.5, model_config['data']['y_range'][1] + 0.5])

    plt.tight_layout()
    plt.savefig(f'{figures_dir}/measurement_histograms.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measurement histograms experiment')
    parser.add_argument('protocol', type=str, help='Protocol name (e.g. measurement_histograms)')
    args = parser.parse_args()

    df, model_name = run(args.protocol)
    plot(df, model_name, args.protocol)
