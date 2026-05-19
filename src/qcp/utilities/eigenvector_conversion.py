import torch
import numpy as np

def eigenstate_to_value(input_data, num_qubits, min_val, max_val):

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


def value_to_eigenstate(values, n_qubits, min_val, max_val):
    
    statevectors = np.round((values - min_val) * (2**n_qubits - 1) / (max_val - min_val))
    
    return statevectors

def eigenstate_to_bits(denary_number, n_bits):
    
    binary_string = bin(denary_number)[2:].zfill(n_bits)
    
    return [int(bit) for bit in binary_string]
