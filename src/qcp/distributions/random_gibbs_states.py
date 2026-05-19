# Third-party imports
import numpy as np
import scipy

from torch import tensor

class RandomGibbsStates:

    def __init__(self, num_classes, num_features, dimension, density, temperature):

        self.num_classes = num_classes
        self.num_features = num_features
        self.dimension = dimension
        self.density = density
        self.temperature = temperature
    

    def generate_gibbs_state(self, hamiltonian_matrix):
        """
        Generate the Gibbs state from a given Hamiltonian matrix.

        This function first ensures that the provided matrix is Hermitian,
        then computes its exponential weighted by the inverse temperature (beta = 1/T),
        and finally normalizes it to form a valid density matrix (rho).

        Parameters:
            hamiltonian_matrix (ndarray):
                A matrix (complex-valued) representing the Hamiltonian of the system. This matrix
                is converted to a Hermitian form prior to exponentiation.

        Returns:
            ndarray:
                The normalized density matrix (rho), obtained by exponentiating
                the Hermitian version of the Hamiltonian matrix and dividing by its trace.
        """

        hermitian_matrix = (hamiltonian_matrix + hamiltonian_matrix.T.conj())/2
        beta = 1/self.temperature
        
        exponential_hermitian = scipy.linalg.expm(-beta * hermitian_matrix)
        trace = np.trace(exponential_hermitian) # get the trace
        rho = exponential_hermitian / trace # normalize

        return rho

    def generate_class_label_from_gibbs_state(self, rho):
        """
        Compute a discrete class label from a given Gibbs state using von Neumann entropy.
        
        Parameters:
            rho (ndarray):
                The density matrix (Gibbs state) for which we wish to derive a class label.

        Returns:
            int:
                An integer class label derived from the normalized von Neumann entropy
                (ranging between 0 and num_classes-1).
        """

        eigenvalues = np.linalg.eigvalsh(rho)
        non_zero_eigenvalues = eigenvalues[eigenvalues > 1e-14]
        entropy = -np.sum(non_zero_eigenvalues * np.log(non_zero_eigenvalues))

        entropy_score = entropy / np.log(self.dimension)

        class_label = int(entropy_score * (self.num_classes - 1))       

        return class_label

    def generate_data(self):
        """
        Generate Hamiltonian matrices and associated class labels for training or analysis.

        Returns:
            (torch.Tensor, torch.Tensor):
                A tuple of tensors:
                    - The first element is a tensor of shape (num_features, dimension, dimension),
                      containing the random Hamiltonian matrices.
                    - The second element is a 1D tensor of length num_features, containing
                      the integer class labels.
        """

        hamiltonian_matrices = []
        classes = []

        for _ in range(self.num_features):

            # Create a Hamiltonian matrix and make it Hermitian
            hamiltonian_matrix = scipy.sparse.random(self.dimension, self.dimension, density=self.density, format='csr', dtype=np.complex128).toarray()
            hamiltonian_matrices += [hamiltonian_matrix]

            # Compute the Gibbs state
            gibbs_state = self.generate_gibbs_state(hamiltonian_matrix)

            # Normalize and discretize the Gibbs state to work with the PQC
            state_class = self.generate_class_label_from_gibbs_state(gibbs_state)
            classes += [state_class]

        hamiltonian_matrices = tensor(np.array(hamiltonian_matrices))
        classes = tensor(np.array(classes))

        return (hamiltonian_matrices, classes)


    
    
