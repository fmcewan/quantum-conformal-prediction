import sqlite3
import json
import mthree

from qiskit_ibm_runtime.fake_provider import FakeQuitoV2

from qcp.utilities.file_handling import load_yaml
from qcp.models.factory import get_circuit
from qcp.utilities.eigenvector_conversion import eigenstate_to_value

DB_PATH = "./data/jobs/jobs.db"

class CircuitManager:

    def __init__(self, name: str, hardware: str) -> None:
        path = f'./data/models/{name}/configuration.yml'

        self.training_configuration = load_yaml(path)
        self.model_configuration = self.training_configuration['model']

        self.name = name
        self.hardware = hardware
        self.type = self.training_configuration['model']['type']
        self.y_range = self.training_configuration['data']['y_range']
        self.n_qubits = self.training_configuration['model']['wires']
        
        self.circuit, self.angle_encoder = get_circuit(self.name, self.type, self.model_configuration)
        
        self.data = {}
        self.data_binary = {}

    def set_hardware(self, new_hardware: str) -> None:
        self.hardware = new_hardware
        self.circuit, self.angle_encoder = get_circuit(self.name, self.type, self.model_configuration)

    def extract_shots(self, job_id: str, M: int) -> None:

        match self.hardware:
            case "aer":
                self.data_binary = self._load_from_db(job_id)

                if self.type == "classification":
                    self.data = {int(k, 2): v for k, v in self.data_binary.items()}

                else:
                    self.data = eigenstate_to_value(
                        self.data_binary, self.n_qubits, self.y_range[0], self.y_range[1]
                    ) 

            case "ibmq":
                self.data_binary = self._load_from_db(job_id)
                self.data = eigenstate_to_value(
                    self.data_binary, self.n_qubits, self.y_range[0], self.y_range[1]
                )

            case "ibmqM3":
                counts = self._load_from_db(job_id)
                backend = FakeQuitoV2()
                mit = mthree.M3Mitigation(backend)
                mit.cals_from_system(range(self.n_qubits))
                m3_quasis = mit.apply_correction(counts, range(self.n_qubits))
                probabilities = m3_quasis.nearest_probability_distribution()
                self.data_binary = {
                    bit_string: round(M * prob)
                    for bit_string, prob in probabilities.items()
                }
                self.data = eigenstate_to_value(
                    self.data_binary, self.n_qubits, self.y_range[0], self.y_range[1]
                )

            case _:
                raise ValueError(f"Unsupported hardware type: {self.hardware}")

    def _load_from_db(self, job_id: str) -> dict:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT counts FROM shots WHERE job_id = ?", (job_id,)
        ).fetchone()
        conn.close()

        if row is None:
            raise KeyError(f"Job ID '{job_id}' not found in database.")

        return json.loads(row[0])
