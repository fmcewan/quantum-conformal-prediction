# Standard library imports 
import csv
import os

# Third-party imports
import numpy as np
import torch
import yaml

# Qiskit imports 
from qiskit_ibm_runtime.fake_provider import FakeQuitoV2
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import Batch
from qiskit_aer import AerSimulator

# Local module imports
from qcp.models.circuits.circuit_manager import CircuitManager 
from qcp.distributions.distribution_manager import create_distribution

# Constant for the YAML file where Aer counts are saved.
AER_SAVE_DIRECTORY = "./data/jobs/aer_counts.yml"
IBMQ_SAVE_DIRECTORY = "./saved/jobs/ibmq_counts.yml"
IBMQM3_SAVE_DIRECTORY = "./saved/jobs/ibmqM3_counts.yml"

def unsupervised_aer_jobs(model, n_jobs, M):
    
    os.makedirs("./data/jobs", exist_ok=True)
    
    try:
        with open(AER_SAVE_DIRECTORY, 'r') as file:
            aer_data = yaml.safe_load(file)
    except FileNotFoundError:
        aer_data = {}
    if aer_data is None:
        aer_data = {}

    job_ids = []

    for i in range(n_jobs):
        job_id = f"aer_{model.name}_{len(aer_data) + i}"
        counts = model.circuit.sample_from_model(M)
        job_ids.append(job_id)
        aer_data[job_id] = counts

    with open(AER_SAVE_DIRECTORY, 'w') as file:
        yaml.dump(aer_data, file)

    return job_ids

def unsupervised_ibmq_jobs(model, backend, n_jobs, M):
   
    try:
        with open(IBMQ_SAVE_DIRECTORY, 'r') as ibmq_file:
            ibmq_data = yaml.safe_load(ibmq_file)
    except FileNotFoundError:
        ibmq_data = {}
    if ibmq_data is None:
        ibmq_data = {}

    try:
        with open(IBMQM3_SAVE_DIRECTORY, 'r') as ibmqM3_file:
            ibmqM3_data = yaml.safe_load(ibmqM3_file)
    except FileNotFoundError:
        ibmqM3_data = {}
    if ibmqM3_data is None:   
        ibmqM3_data = {}

    job_ids = []
    sampler = Sampler(mode=backend)

    for _ in range(n_jobs):
        job = backend.run([model.circuit], shots=M)
        job_id = job.job_id()
        job_ids.append(job_id)
        ibmq_data[job_id] = dict(job.result().get_counts())
        ibmqM3_data[job_id] = dict(job.result().get_counts())

    with open(IBMQ_SAVE_DIRECTORY, 'w') as ibmq_file:
        yaml.dump(ibmq_data, ibmq_file)
    with open(IBMQM3_SAVE_DIRECTORY, 'w') as ibmqM3_file:
        yaml.dump(ibmqM3_data, ibmqM3_file)

    return job_ids


def supervised_aer_jobs(model, x_points, M):

    try:
        with open(AER_SAVE_DIRECTORY, 'r') as file:
            aer_data = yaml.safe_load(file)
    except FileNotFoundError:
        aer_data = {}
    if aer_data is None:
        aer_data = {}

    simulator = AerSimulator(method='statevector')

    bound_isa_pqcs = []
    for x_point in x_points:
        x_tensor = torch.FloatTensor([x_point]) 
        angles = model.angle_encoder(x_tensor)
        angles_np = angles.detach().numpy()
        bound_isa_pqcs.append(model.circuit.assign_parameters(angles_np.flatten(), inplace=False))

    job_ids = []
    for bound_isa_pqc in bound_isa_pqcs:
        job = simulator.run(bound_isa_pqc, shots=M)
        job_id = job.job_id()
        job_ids.append(job_id)
        aer_data[job_id] = dict(job.result().get_counts())

    with open(AER_SAVE_DIRECTORY, 'w') as file:
        yaml.dump(aer_data, file)

    return job_ids


def supervised_ibmq_jobs(model, backend, x_points, M):

    try:
        with open(IBMQ_SAVE_DIRECTORY, 'r') as file:
            ibmq_data = yaml.safe_load(file)
    except FileNotFoundError:
        ibmq_data = {}
    if aer_data is None:
        ibmq_data = {}

    bound_isa_pqcs = []
    for x_point in x_points:
        x_tensor = torch.FloatTensor([x_point]) 
        angles = model.angle_encoder(x_tensor)
        angles_np = angles.detach().numpy()
        bound_isa_pqcs.append(model.circuit.assign_parameters(angles_np.flatten(), inplace=False))

    job_ids = []
    for bound_isa_pqc in bound_isa_pqcs:
        job = simulator.run(bound_isa_pqc, shots=M)
        job_id = job.job_id()
        job_ids.append(job_id)
        aer_data[job_id] = dict(job.result().get_counts())
    
    with open(IBMQ_SAVE_DIRECTORY, 'w') as file:
        yaml.dump(ibmq_data, file)

    return job_ids

def classification_aer_jobs(model, x_points, M):

    try:
        with open(AER_SAVE_DIRECTORY, 'r') as file:
            aer_data = yaml.safe_load(file)
    except FileNotFoundError:
        aer_data = {}
    if aer_data is None:
        aer_data = {}

    simulator = AerSimulator(method='statevector')
    
    bound_isa_pqcs = []   
    for x in x_points:
        real_part = x.real.reshape(-1)
        imag_part = x.imag.reshape(-1)
        x = torch.cat((real_part, imag_part)).float().unsqueeze(0)
        
        angles = model.angle_encoder(x)
        angles_np = angles.detach().numpy()
        bound_isa_pqcs.append(model.circuit.assign_parameters(angles_np[0], inplace=False))

    job_ids = []
    with Batch(backend=simulator):
        for bound_isa_pqc in bound_isa_pqcs:
            job = simulator.run(bound_isa_pqc, shots=M)
            job_id = job.job_id()
            job_ids.append(job_id)
            aer_data[job_id] = dict(job.result().get_counts())

    # Write the updated job data back to the YAML file.
    with open(AER_SAVE_DIRECTORY, 'w') as file:
        yaml.dump(aer_data, file)

    return job_ids

def classification_ibmq_jobs(model, backend, x_points, M):

    job_ids = []
    backend = FakeQuitoV2()

    # Convert x_points into a Torch FloatTensor for processing.
    bound_isa_pqcs = []   
    for x in x_points:
        real_part = x.real.reshape(-1)
        imag_part = x.imag.reshape(-1)
        x = torch.cat((real_part, imag_part)).float().unsqueeze(0)
        
        angles = model.angle_encoder(x)
        angles_np = angles.detach().numpy()
        bound_isa_pqcs.append(model.circuit.assign_parameters(angles_np[0], inplace=False))
    
    job_ids = []
    with Batch(backend=backend):
        sampler = Sampler(mode=backend)
        for bound_isa_pqc in bound_isa_pqcs:
            job = sampler.run([bound_isa_pqc], shots=M)
            job_id = job.job_id()
            job_ids.append(job_id)
            aer_data[job_id] = dict(job.result().get_counts())

    return job_ids

def run_and_save_jobs(hardware, model_name, n_jobs, M):

    backend = FakeQuitoV2()
    model = CircuitManager(model_name, hardware)
    
    if model.type == "unsupervised":
        data_config = model.training_configuration["data"]
        distribution = create_distribution(data_config)

        x_points = np.full(shape=n_jobs, fill_value=-1, dtype=int)
        y_points = distribution.rvs(size=n_jobs)
    
    elif model.type == "supervised":
        data_config = model.training_configuration["data"]
        distribution = create_distribution(data_config)

        x_points, y_points = distribution.rvs(size=n_jobs)

    elif model.type == "classification":
        data_config = model.training_configuration["data"]
        data_config["num_features"] = n_jobs
        
        distribution = create_distribution(data_config)
        
        x_points, y_points = distribution.generate_data()

    if hardware == "aer" and model.type == "unsupervised":
        job_ids = unsupervised_aer_jobs(model, n_jobs, M)

    elif hardware in ("ibmq", "ibmqM3") and model.type == "unsupervised":
        job_ids = unsupervised_ibmq_jobs(model, backend, n_jobs, M)
    
    elif hardware == "aer" and model.type == "supervised":
        job_ids = supervised_aer_jobs(model, x_points, M)
    
    elif hardware in ("ibmq", "ibmqM3") and model.type == "supervised":
        job_ids = supervised_ibmq_jobs(model, backend, x_points, M)
    
    elif hardware == "aer" and model.type == "classification":
        job_ids = classification_aer_jobs(model, x_points, M)
    
    elif hardware in ("ibmq", "ibmqM3") and model.type == "classification":
        job_ids = classification_ibmq_jobs(model, backend, x_points, M)
    
    else:
        raise NotImplementedError

    file_name = f"data/jobs/{hardware}_{model.name}_M{M}.csv"
    
    with open(file_name, 'w', newline='') as csvfile:

        job_writer = csv.writer(csvfile, delimiter=',')
        job_writer.writerow(["x", "y", "job_id"])
        
        for x, y, job_id in zip(x_points, y_points, job_ids):
            job_writer.writerow([x, y, job_id])

