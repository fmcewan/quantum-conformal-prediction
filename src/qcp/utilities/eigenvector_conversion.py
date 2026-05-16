import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt  
from scipy.stats import norm

def evenly_space_eigenstates(input_data, num_qubits, min_val, max_val):
    """
    Converts eigenstates in either denary or in the computational basis to a value in the output range.
    Evenly spaces eigenvectors across the range.
    Supports single values, lists, and tensors of strings or integers.
    """

    N = 2 ** num_qubits

    def convert_to_output_space(index):
        return min_val + (index) * (max_val - min_val) / (N - 1)

    def process_bitstring(bitstring):
        return convert_to_output_space(int(bitstring, 2))

    if isinstance(input_data, str):
        return process_bitstring(input_data)
    elif isinstance(input_data, int):
        return convert_to_output_space(input_data)
    elif isinstance(input_data, dict):
        return {process_bitstring(key): value for key, value in input_data.items()}
    elif isinstance(input_data, (list, np.ndarray)):
        if isinstance(input_data[0], str):
            return [process_bitstring(bs) for bs in input_data]
        return [convert_to_output_space(i) for i in input_data]
    elif isinstance(input_data, torch.Tensor):
        return torch.tensor([convert_to_output_space(i.item()) for i in input_data.flatten()]).view(input_data.size())
    else:
        raise TypeError(f"Unsupported input type: {type(input_data)}")


def to_closest_eigenstate(values, n_qubits, min_val, max_val, base=2):
    statevectors = np.round((values - min_val) * (2**n_qubits - 1) / (max_val - min_val))
    return statevectors

def denary_eigenstate_to_binary_array(denary_number, n_bits):
    binary_string = bin(denary_number)[2:].zfill(n_bits)
    return [int(bit) for bit in binary_string]


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
