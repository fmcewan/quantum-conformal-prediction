 # MUST BE IN TQVENV
    def plot_training_results(self, trained_pqc):

        from sklearn.neighbors import KernelDensity
        from utils.eigenvector_conversion import evenly_space_eigenstates 
        from scipy.stats import norm
        from qiskit.visualization import plot_histogram

        # measure with torchquantum simulator
        measurements = trained_pqc.sample_from_model(1000)
        data = measurements[0]

        # create axes
        _, ax1 = plt.subplots()
        x_values = np.linspace(self.y_range[0], self.y_range[1], 1000)
        ax2 = ax1.twinx()
        
        # plot histogram of measured states
        plot_histogram(data)
        states = [evenly_space_eigenstates(bitstring, self.n_qubits, self.y_range[0], self.y_range[1]) for bitstring in data.keys()]
        state_frequencies = list(data.values())
        
        ax1.bar(states, state_frequencies, label='Histogram of Measurements')

        # plot kernel density estimation of measured states
        expanded_data = []
        for key, freq in data.items():
            value = evenly_space_eigenstates(key, self.n_qubits, self.y_range[0], self.y_range[1])
            expanded_data.extend([value] * freq)
        expanded_data = np.array(expanded_data).reshape(-1, 1)
        kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(np.array(expanded_data).reshape(-1, 1))
        kde_values = np.exp(kde.score_samples(x_values.reshape(-1, 1)))
        ax2.plot(x_values, kde_values, color="r", label="KDE of Measurements")

        true_distribution = create_distribution(self.config['data'])
        x_points = np.linspace(self.y_range[0], self.y_range[1], 1000)
        y_points = [true_distribution._pdf(x_point) for x_point in x_points]
        ax2.plot(x_points, y_points, color='black', label='True distribution')

        eigenvalues = evenly_space_eigenstates(torch.arange(start=0, end= 2**self.n_qubits, step=1), self.n_qubits, self.y_range[0], self.y_range[1])
        exp_val = trained_pqc.calculate_expected_value(eigenvalues).item()
        ax2.axvline(x=exp_val, color='r', linestyle='--', label=f'expected_value = {exp_val}')

        plt.show()


    # MUST BE IN TQVENV
    def plot_training_results_reg2(self, trained_pqc):
        
        from distributions.heteroscedastic import HeteroscedasticData
        
        # measure with torchquantum simulator
        x_points = np.linspace(self.x_range[0], self.x_range[1], 300)    
        samples = trained_pqc.sample_from_model(torch.from_numpy(x_points))

        # create axes
        _, ax1 = plt.subplots()
        x_positive = np.linspace(self.x_range[0], self.x_range[1], 500)
        
        # plot histogram of measured states
        ax1.scatter(x_points, samples, label='Scatter of Measurements')

        # plot true curve
        dist = HeteroscedasticData([self.x_range[0], self.x_range[1]])
        true_x_samples, true_y_samples = dist.rvs(size=500)
        ax1.scatter(true_x_samples, true_y_samples, color="green")   
        
        continuous_y2 = dist.component_mean(x_positive)
        ax1.plot(x_positive, continuous_y2, color='g', label="True Distribution")

        plt.show()

    def plot_training_results_reg1(self, trained_pqc):

        # measure with torchquantum simulator
        x_points = np.linspace(self.x_range[0], self.x_range[1], 300)    
        samples = trained_pqc.sample_from_model(torch.from_numpy(x_points))

        # create axes
        fig, ax1 = plt.subplots()
        continuous_x = np.linspace(self.x_range[0], self.x_range[1], 1000)
        
        # plot histogram of measured states
        ax1.scatter(x_points, samples, label='Scatter of Measurements')

        # plot true curve
        continuous_y = 0.5*np.sin(0.8*continuous_x) + 0.05*continuous_x
        ax1.plot(continuous_x, continuous_y, color='g', label="True Distribution")
        ax1.plot(continuous_x, -continuous_y, color='g', label="True Distribution")

        plt.show()

def plot_training_results_classification(self, trained_pqc):

    distribution = create_distribution("classification")
    features, true_labels = distribution.generate_data()
    
    predicted_labels = trained_pqc.sample_from_model(features)

    # Convert to numpy arrays for convenience
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)
    cm = confusion_matrix(true_labels, predicted_labels)

    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    num_classes = cm.shape[0]
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="red", fontsize=10
            )
    fig.colorbar(im, ax=ax)

    ax2 = axes[1]
    ax2.scatter(true_labels, predicted_labels, alpha=0.6)
    ax2.set_title("Predicted vs. True Labels")
    ax2.set_xlabel("True Labels")
    ax2.set_ylabel("Predicted Labels")
    min_lab = min(true_labels.min(), predicted_labels.min())
    max_lab = max(true_labels.max(), predicted_labels.max())
    ax2.plot([min_lab, max_lab], [min_lab, max_lab], 'r--', lw=2)

    fig.suptitle(f"Classification Results\nAccuracy = {accuracy*100:.2f}%")
    plt.tight_layout()
    plt.show()
