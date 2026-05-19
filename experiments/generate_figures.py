# Default imports
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches
import numpy as np
import os
import yaml
import math

# File imports
from conformal_prediction.cp_algorithm import CPAlgorithm 
from utils.file_handling import load_yaml
from utils.eigenvector_conversion import to_closest_eigenstate
from distributions.dist_manager import create_distribution, event_probability
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import beta, betabinom
from scipy.optimize import brentq
import itertools
import seaborn as sns

base_size = 9
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': base_size,
    'axes.titlesize': base_size + 2,
    'axes.labelsize': base_size,
    'xtick.labelsize': base_size - 2,
    'ytick.labelsize': base_size - 2,
    'legend.fontsize': base_size - 1,
    'figure.titlesize': base_size +4
})

inside_title_args = {
    'fontsize': 9,
    'va': 'top',
    'ha': 'center',
    'color': 'red', 
    'bbox': {'facecolor': 'white', 'edgecolor': 'white', 'alpha': 0.0}
}

data_config_unsup_1 = {
    "type": "function",
    "distribution": "combined_normals",
    "y_range": [-1.5, 1.5],
    "component_means": [-0.75, 0.75],
    "component_stds": [0.1, 0.1]
}

hardware_styles = {
        "aer_knn": {"color": "black", "linestyle": "-", "set_size_color": "green"},
        "ibmq_knn": {"color": "orange", "linestyle": "--", "set_size_color": "green"},
        "ibmqM3_knn": {"color": "blue", "linestyle": "-.", "set_size_color": "green"}
}

algorithm_colours = ['#27CFE8', '#2755E8', '#2791E8', '#1BEBC3', '#91C3EB']

TRUE_DIST_SET_COLOUR = ""
TRUE_DIST_SAMPLES_COLOUR = ""
PREDICTION_SET_COLOUR = ""
MODEL_SAMPLES_COLOUR = ""

def true_distribution(figure_id):
    """plot of pdf with high density region shading"""
    
    # Initialise plot figures
    fig, ax = plt.subplots()
    plt.subplots_adjust(hspace=0.5)
    
    # Initialise figure and data configurations
    common_properties = load_yaml(f"saved/figures/figure_{figure_id}/config.yaml")['common_properties']
    data_configuration = load_yaml(f"saved/models/{common_properties['model_name']}/config.yaml")['data']

    # Create distribution and parameters
    true_probability_distribution = create_distribution(data_configuration) 

    # Generating PDF and plotting the true distribution
    x_points = np.linspace(-3, 4, 1000)
    y_points = []

    for x_point in x_points:
        y_point = true_probability_distribution._pdf(x_point)
        y_points += [y_point]
    
    # Plot the true distribution
    ax.plot(x_points, y_points, color='black') 

    plt.savefig(f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf")
    plt.show()

# Model architechture
def model_architechture(figure_id):
    """diagram of model architechture"""

        

# Trained models
def measurement_histograms(figure_id):
    """
    Plots of the measurements from the trained PQC's probability distribution on a classical simulator, 
    an IBMQ device and an IMBQ device with M3 QEM, as well as plot of the true distribution of the PQC
    
    arguments:
    figure_id: id for figure folder
    
    -- the figure folder should have a results.csv file in it, with pre-generated data based on what needs to be shown 
    """
    
    # Initialise plot figures 
    figure, axes = plt.subplots(2,2)
   
    # Flatten axes
    axes = axes.flatten()

    # Initialise figure and data configurations
    common_properties = load_yaml(f"saved/figures/figure_{figure_id}/config.yaml")['common_properties']
    data_configuration = load_yaml(f"saved/models/{common_properties['model_name']}/config.yaml")['data']
    
    # Create distribution and parameters
    true_probability_distribution = create_distribution(data_configuration)

    # Generating PDF and plotting true distribution
    x_points = np.linspace(-1.5, 1.5, 1000)
    y_points = []

    for x_point in x_points:
        y_point = true_probability_distribution._pdf(x_point)
        y_points += [y_point]

    # Plot X and Y axes for the Ground Truth Distribution
    axes[0].plot(x_points, y_points, color='black')

    # Generating and plotting data for histograms of data from the PQC
    df = pd.read_csv(f"saved/figures/figure_{figure_id}/results.csv")

    for idx, column in enumerate(df.columns[2:], start=0):
        ax = axes[idx+1]
        ax.bar(df['y_values'], df[column], ec='none', alpha=0.5, width=0.12)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    border_color = 'blue'
    text_color = 'red'

    for ax in axes:
        ax.set_xlim([-2, 2])

    axes[0].text(0.5, 0.95, "Ground Truth Density Function", transform=axes[0].transAxes, **inside_title_args)
    axes[0].set_xticks([])

    axes[1].text(0.5, 0.95, "PQC Histogram\n(classical simulator)", transform=axes[1].transAxes, **inside_title_args)
    axes[1].set_xticks([])

    axes[2].text(0.5, 0.95, "PQC Histogram\n(IBMQ NISQ Device Simulator)", transform=axes[2].transAxes, **inside_title_args)
    axes[2].set_xlabel("y")

    axes[3].text(0.5, 0.95, "PQC Histogram\n(IBMQ NISQ Device Simulator with M3 QEM)", transform=axes[3].transAxes, **inside_title_args)
    axes[3].set_xlabel("y")
    
    plt.savefig(f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf")
    plt.show()

# Unsupervised prediction experiments

def set_size_and_coverage(figure_id):

    """
    Generates and saves a dual-axis plot for coverage and set size vs. epsilon.

    Args:
        csv_filepath (str): The path to the CSV file containing the results.
        figure_id (str): Identifier for the folder where the plot will be saved.
        hardware_options (dict): A dictionary to map algorithm names to plot styles.
                                 e.g., {'aer_knn': {'color': 'blue', 'linestyle': '-'}, ...}
    """

    csv_filepath = f"saved/figures/figure_{figure_id}/results.csv"

    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_filepath}")
        return

    # Create the plot and the first (left) y-axis for coverage
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Significance Level (α)')
    ax1.set_ylabel('Coverage', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, which='both', linestyle='-', linewidth=1)

    # Plot the target coverage line (y = 1 - alpha)
    ax1.plot(df['alpha'].unique(), 1 - df['alpha'].unique(), 'r-', label='Target Coverage (1-α)')

    # Create the second (right) y-axis for Set Size
    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Set Size', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Group data by algorithm and plot the lines
    for name, group in df.groupby('algorithm'):
        style = hardware_styles.get(name, {'color': 'black', 'linestyle': '-'}) # Default style
        
        # Plot achieved coverage on the left axis
        ax1.plot(group['alpha'], group['avg_coverage'], 
                 label=f'{name} Coverage', 
                 color=style['color'], 
                 linestyle=style['linestyle'])

        # Plot average set size on the right axis
        ax2.plot(group['alpha'], group['avg_set_size'], 
                 label=f'{name} Set Size', 
                 color=style.get('set_size_color', 'green'), # Use a different color for set size
                 linestyle=style['linestyle'])

    # Final plot adjustments
    fig.suptitle('Coverage and Set Size vs. Significance Level', fontsize=16)
    fig.legend(loc="upper right", bbox_to_anchor=(0.94, 0.861))
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save the figure
    plt.savefig(f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf")
    plt.show()

def unsupervised_informativeness(figure_id):
    """
    Generates and saves a bar chart comparing the informativeness (average set size)
    of different models for the unsupervised task.

    This function reads a pre-generated CSV file containing the average set size
    and standard deviation for each model quality type ("Good Model", "Bad Model").

    Args:
        figure_id (str): Identifier for the folder where the results CSV is located
                         and where the plot will be saved.
    """
    # Construct the file path internally using the figure_id
    csv_filepath = f"saved/figures/figure_{figure_id}/results.csv"
    
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_filepath}")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Define colors for the bars
    colors = ['#0072B2', '#D55E00']  # A nice blue and orange

    # Create the bar chart with error bars
    bars = ax.bar(
        df['model_quality'],
        df['avg_set_size'],
        yerr=df['set_size_std'],  # Use the standard deviation for error bars
        capsize=5,               # Adds caps to the error bars
        color=colors,
        alpha=0.8
    )

    # Add labels and title for clarity
    ax.set_ylabel('Average Prediction Set Length', fontsize=12)
    ax.set_title('Informativeness', fontsize=14, pad=20)
    ax.set_ylim(bottom=0) # Ensure the y-axis starts at 0
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + 0.05,  # Position text slightly above the bar
            f'{yval:.2f}',
            ha='center', 
            va='bottom',
            fontsize=11
        )
        
    # Add a text box explaining the coverage
    # This is crucial to show it's a fair comparison
    coverage_text = f"Note: Both models achieved the target {1-0.05:.0%} coverage"
    fig.text(0.5, 0.01, coverage_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make room for text
    
    # Save the figure
    output_path = f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf"
    plt.savefig(output_path)
    plt.show()

# Supervised prediction experiments 

def prediction_intervals(figure_id):
    """
    Generates and saves a plot of the prediction intervals for the
    supervised regression experiment.

    Args:
        figure_id (str): Identifier for the folder where the results CSV is located
                         and where the plot will be saved.
    """

    csv_filepath = f"saved/figures/figure_{figure_id}/results.csv"
    
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_filepath}")
        return

    # Sorting the dataframe by the x-values for correct line plotting
    df = df.sort_values(by='x').dropna()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['x'], df['y_true'], color='black', s=10, alpha=0.5, label='True Data')
    ax.plot(df['x'], df['y_hat'], color='red', linestyle='--', linewidth=2, label='Model Prediction ($\hat{y}$)')

    # Plotting the prediction interval as a shaded region
    ax.fill_between(
        df['x'],
        df['y_lower'],
        df['y_upper'],
        color='red',
        alpha=0.2,
        label='95% Prediction Interval'
    )

    # Addijng the labels, title, and legend
    ax.set_xlabel('Input features ($x$ points)', fontsize=12)
    ax.set_ylabel('Output ($\hat{y}$)', fontsize=12)
    ax.set_title('Predictions Intervals for Heteroscedastic Regression', fontsize=14, pad=20)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf"
    plt.savefig(output_path)
    plt.show()

def supervised_informativeness(figure_id):
    """
    Generates and saves a grouped bar chart comparing the informativeness
    of good vs. bad models for classification and regression tasks.
    """
    csv_filepath = f"saved/figures/figure_{figure_id}/results.csv"
    
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_filepath}")
        return

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use seaborn for a clean grouped bar plot
    sns.barplot(
        x='task', 
        y='avg_set_size', 
        hue='model_quality', 
        data=df, 
        palette={'Good Model': '#0072B2', 'Bad Model': '#D55E00'},
        ax=ax
    )

    # Add error bars manually
    # The positions of the bars are at 0 and 1 for the tasks
    # The hue offset will be approximately -0.2 and 0.2 for the two bars
    tasks = df['task'].unique()
    for i, task in enumerate(tasks):
        task_data = df[df['task'] == task]
        good_model_data = task_data[task_data['model_quality'] == 'Good Model']
        bad_model_data = task_data[task_data['model_quality'] == 'Bad Model']
        
        # Add error bar for good model
        if not good_model_data.empty:
            ax.errorbar(x=i - 0.2, y=good_model_data['avg_set_size'], 
                        yerr=good_model_data['set_size_std'], fmt='none', c='black', capsize=5)
        # Add error bar for bad model
        if not bad_model_data.empty:
            ax.errorbar(x=i + 0.2, y=bad_model_data['avg_set_size'], 
                        yerr=bad_model_data['set_size_std'], fmt='none', c='black', capsize=5)

    # Add labels and title
    ax.set_ylabel('Average Prediction Set Size / Length', fontsize=12)
    ax.set_xlabel('Supervised Task', fontsize=12)
    ax.set_title('QCP Informativeness vs. Model Quality (at 95% Coverage)', fontsize=16, pad=20)
    ax.tick_params(axis='x', labelsize=12)
    
    # Improve legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Model Quality', fontsize=11, title_fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = f"saved/figures/figure_{figure_id}/figure_{figure_id}.pdf"
    plt.savefig(output_path)
    plt.show()


# Classification prediction experiments 

# Plots of the set size against M 
def set_size_against_m(figure_id):
    """plot the average set size of conformal algorithms over varied M"""

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    axes[0].set_xlabel("number of quantum measurements (M)")
    axes[0].set_ylabel("average size of the set predictor")
    axes[0].set_xscale('log')
    axes[0].set_title('model')

    axes[1].set_xlabel("number of quantum measurements (M)")
    axes[1].set_ylabel("average coverage of the set predictor")
    axes[1].set_xscale('log')
    axes[1].set_title('model')

    results_config = load_yaml(f"saved/figures/figure_{figure_id}/config.yaml")
    common_properties = results_config['common_properties']
    data_config = load_yaml(f"saved/models/{common_properties['model_name']}/config.yaml")['data']

    true_distribution = create_distribution(data_config)
    true_quantile_ranges = true_distribution.get_hdr(1-common_properties['alpha'], [data_config['y_range'][0], data_config['y_range'][1]])
    true_hdr_size = sum(interval[1] - interval[0] for interval in true_quantile_ranges)

    axes[0].axhline(y=true_hdr_size, color='grey', linestyle='--', label='Gold Standard Set Size')
    M_list = common_properties['M_list']
    dfs = [pd.read_csv(f"saved/figures/figure_{figure_id}/results_M{M}.csv") for M in M_list]
    set_size_cols = dfs[0].columns[1::2]
    coverage_cols = dfs[0].columns[2::2]

    average_set_sizes = {col: [df[col].mean() for df in dfs] for col in set_size_cols}
    average_coverages = {col: [df[col].mean() for df in dfs] for col in coverage_cols}

    for idx, col in enumerate(set_size_cols):
        axes[0].scatter(M_list, average_set_sizes[col], color=algorithm_colours[idx], label=col)
        axes[0].plot(M_list, average_set_sizes[col], linestyle='-', color=algorithm_colours[idx])

    axes[1].axhline(y=0.9, color='grey')
    for idx, col in enumerate(coverage_cols):
        axes[1].scatter(M_list, average_coverages[col], color=algorithm_colours[idx])
        axes[1].plot(M_list, average_coverages[col], linestyle='-', color=algorithm_colours[idx])

    fig.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.0, 1.0))

    # ax.set_ylim(bottom=0)

    plt.savefig(f"saved/figures/figure_{figure_id}/figure.pdf")


def coverage_against_m(figure_id):
    """plot the average coverage of conformal algorithms over varied M"""

    fig, ax = plt.subplots(1, figsize=(12, 4))

    ax.set_xlabel("number of quantum measurements (M)")
    ax.set_ylabel("average coverage set predictor")
    ax.set_xscale('log')

    df = pd.read_csv(f"saved/figures/figure_{figure_id}/results.csv")

    for _, column in enumerate(df.columns[1:]):
        ax.plot(df['M'], df[column], linestyle='-', color='#41B6FA', alpha=0.8)

    ax.legend()
    ax.relim()
    ax.autoscale()  
    # ax.set_ylim(bottom=0)

    plt.savefig(f"saved/figures/figure_{figure_id}/figure.pdf")

def coverage_distribution_ideal(figure_id):

    ns = [100,1000,10000]
    alpha = 0.1

    sns.set_palette('pastel')
    plt.figure()
    ax = plt.gca()

    for i in range(len(ns)):
        n = ns[i]
        l = np.floor((n+1)*alpha)
        a = n + 1 - l
        b = l
        x = np.linspace(0.825,0.975,1000)
        rv = beta(a, b)
        ax.plot(x, rv.pdf(x), lw=3, label=f'n={n}')
    ax.vlines(1-alpha,ymin=0,ymax=150,color='#888888',linestyles='dashed',label=r'$1-\alpha$')
    sns.despine(top=True,right=True)
    plt.yticks([])
    plt.legend()
    plt.title('Distribution of coverage (infinite validation set)')
    plt.tight_layout()
    plt.show()

def empirical_coverage_distribution_ideal(figure_id):
    n = 1000
    nprimes = [100,1000,10000,100000]
    alpha = 0.1

    sns.set_palette('pastel')
    plt.figure()
    ax = plt.gca()

    # Beta parameters
    l = np.floor((n+1)*alpha)
    a = n + 1 - l
    b = l

    for i in range(len(nprimes)):
        nprime = nprimes[i]
        x = np.array(range(int(nprime*(0.75)),nprime))
        rv = betabinom(nprime,a, b)
        ax.plot(x/nprime, rv.pmf(x) * nprime/nprimes[0], lw=3, label=r"$n_{\rm val}$" + f'={nprime}')

    ax.vlines(1-alpha,ymin=0,ymax=0.45,color='#888888',linestyles='dashed',label=r'$1-\alpha$')
    sns.despine(top=True,right=True)
    plt.yticks([])
    plt.legend()
    plt.title(r"Distribution of coverage with $n_{\rm val}$ validation points ($n=1000$)")
    plt.tight_layout()
    plt.show()

def average_empirical_coverage_distribution_ideal(figure_id):
    # parameters
    alpha = 0.1
    ns = [50]
    nprimes = [50]
    params = [(n,nprime) for n in ns for nprime in nprimes]
    Rs = [100, 1000, 10000, 100000]
    T = 1000
    list_distributions = []
    for R in Rs:
        distributions = []
        for param in params:
            n = param[0]
            nprime = param[1]
            l = np.floor((n+1)*alpha)
            a = n + 1 - l
            b = l

            distribution = np.zeros((T,))
            for t in range(T):
                samples = betabinom.rvs(nprime, a, b, size=R)/nprime
                distribution[t] = samples.mean()
            distributions = distributions + [distribution,]
        list_distributions = list_distributions + [distributions,]

    # Plot
    toPlots = [0]
    for toPlot in toPlots:
        plt.figure()
        ax = plt.gca()
        sns.set_palette('pastel')
        for i in range(len(Rs)):
            distributions = list_distributions[i]
            ax.hist(distributions[toPlot],label=r'$R=$'+str(Rs[i]))
        sns.despine(ax=ax,top=True, right=True)
        plt.title(r"Distribution of $\overline{C}$ with $n=$"+str(params[toPlot][0])+r", $n_{\rm val}=$"+str(params[toPlot][1]), fontsize=14)
        if toPlot == toPlots[-1]:
            plt.legend(fontsize=12)
        plt.tight_layout()
        plt.xticks(fontsize=12)
        plt.yticks([])
        # plt.xlim([0.895,0.905])
        plt.show()

def average_empirical_coverage_distribution(figure_id):

    variable_list = "average_coverage"

    df = pd.read_csv(f'saved/figures/{figure_id}/sample_results.csv')
    data = df[variable_list]

    bins = 50

    # Plot the histogram with 30 bins and black edges for each bar
    plt.hist(data, bins=bins, range=(0.60, 1.0), edgecolor='black')
    plt.axvline(x=data.mean(), color='grey')
    # Add title and labels
    plt.title(f'Histogram of {variable_list}')
    plt.xlabel(f'{variable_list}')
    plt.ylabel(f'Number of results with {variable_list}')

    # Display the plot
    plt.show()
    
def plot_samples_and_distribution(training_data, calibration_data):
    
    min_data_point = min(-1.5, min(training_data), min(calibration_data))
    max_data_point = max(1.5, max(training_data), min(calibration_data))
    x_values = np.linspace(min_data_point, max_data_point, 1000)

    pdf_values = 0.5*(norm.pdf(x_values, loc=-0.75, scale=0.1) + norm.pdf(x_values, loc=0.75, scale=0.1))

    plt.plot(x_values, pdf_values, color='g', label="PDF of Sum of Normals")

    plt.plot(training_data.numpy(), np.zeros_like(training_data.numpy()), 'gx', label="Training Data")
    plt.plot(calibration_data.numpy(), np.zeros_like(calibration_data.numpy()), 'o', markerfacecolor='none', color='blue',label="Calibration Data")

    # Labels and legend
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Samples vs PDF of Distribution')
    plt.legend()

    # Show the plot
    plt.show()

