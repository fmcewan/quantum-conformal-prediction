import csv
import json
import os
import sqlite3
import numpy as np
import torch

from qcp.models.circuits.circuit_manager import CircuitManager
from qcp.distributions.distribution_manager import create_distribution

DB_PATH = "./data/jobs/jobs.db"

def get_connection():
    os.makedirs("./data/jobs", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shots (
            job_id TEXT PRIMARY KEY,
            hardware TEXT NOT NULL,
            model_name TEXT NOT NULL,
            M INTEGER NOT NULL,
            counts TEXT NOT NULL
        )
    """)
    conn.commit()

    return conn


def save_shot(conn, job_id, hardware, model_name, M, counts):
    conn.execute(
        "INSERT OR REPLACE INTO shots (job_id, hardware, model_name, M, counts) VALUES (?, ?, ?, ?, ?)",
        (job_id, hardware, model_name, M, json.dumps(counts))
    )
    conn.commit()


def load_shot(job_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT counts FROM shots WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    
    if row is None:
        raise KeyError(f"Job ID '{job_id}' not found in database.")
    
    return json.loads(row[0])


def unsupervised_aer_jobs(model, n_jobs, M):
    conn = get_connection()
    existing = conn.execute("SELECT COUNT(*) FROM shots WHERE model_name = ?", (model.name,)).fetchone()[0]

    job_ids = []
    for i in range(n_jobs):
        job_id = f"aer_{model.name}_{existing + i}"
        counts = model.circuit.sample_from_model(M)
        save_shot(conn, job_id, 'aer', model.name, M, counts)
        job_ids.append(job_id)

    conn.close()
    
    return job_ids

def supervised_aer_jobs(model, x_points, M):
    conn = get_connection()
    existing = conn.execute("SELECT COUNT(*) FROM shots WHERE model_name = ?", (model.name,)).fetchone()[0]

    job_ids = []
    for i, x_point in enumerate(x_points):
        job_id = f"aer_{model.name}_{existing + i}"
        x_tensor = torch.FloatTensor([x_point])
        counts = model.circuit.sample_from_model(x_tensor, n_shots=M)
        save_shot(conn, job_id, 'aer', model.name, M, counts)
        job_ids.append(job_id)

    conn.close()
    
    return job_ids

def classification_aer_jobs(model, x_points, M):
    conn = get_connection()
    existing = conn.execute("SELECT COUNT(*) FROM shots WHERE model_name = ?", (model.name,)).fetchone()[0]

    job_ids = []
    for i, x in enumerate(x_points):
        job_id = f"aer_{model.name}_{existing + i}"
        counts = model.circuit.sample_from_model(x.unsqueeze(0), n_shots=M)
        save_shot(conn, job_id, 'aer', model.name, M, counts)
        job_ids.append(job_id)

    conn.close()
    
    return job_ids

def run_jobs(hardware, model_name, n_jobs, M):
    model = CircuitManager(model_name, hardware)

    match model.type:
        case "unsupervised":
            distribution = create_distribution(model.training_configuration["data"])
            x_points = np.full(shape=n_jobs, fill_value=-1, dtype=int)
            y_points = distribution.rvs(size=n_jobs)

        case "supervised":
            distribution = create_distribution(model.training_configuration["data"])
            x_points, y_points = distribution.rvs(size=n_jobs)

        case "classification":
            data_config = model.training_configuration["data"].copy()
            data_config["num_features"] = n_jobs
            distribution = create_distribution(data_config)
            x_points, y_points = distribution.generate_data()

        case _:
            raise ValueError(f"Unsupported model type: {model.type}")

    match (hardware, model.type):
        case ("aer", "unsupervised"):
            job_ids = unsupervised_aer_jobs(model, n_jobs, M)
        case ("aer", "supervised"):
            job_ids = supervised_aer_jobs(model, x_points, M)
        case ("aer", "classification"):
            job_ids = classification_aer_jobs(model, x_points, M)
        case (("ibmq" | "ibmqM3"), _):
            raise NotImplementedError("IBMQ support coming soon")
        case _:
            raise ValueError(f"Unsupported hardware/model combination: {hardware}/{model.type}")

    return x_points, y_points, job_ids


def save_jobs(hardware, model_name, x_points, y_points, job_ids, M):
    file_name = f"data/jobs/{hardware}_{model_name}_M{M}.csv"
    
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "job_id"])
        for x, y, job_id in zip(x_points, y_points, job_ids):
            if hasattr(y, 'item'):
                y = y.item()
            writer.writerow([x, y, job_id])


def run_and_save_jobs(hardware, model_name, n_jobs, M):
    x_points, y_points, job_ids = run_jobs(hardware, model_name, n_jobs, M)
    save_jobs(hardware, model_name, x_points, y_points, job_ids, M)
