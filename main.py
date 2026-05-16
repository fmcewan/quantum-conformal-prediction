# Standard library imports
import argparse
import sys

def main():

    # Initialize the main parser
    parser = argparse.ArgumentParser(description="Quantum Conformal Prediction (QCP) Pipeline Manager")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available execution modes")

    # TRAIN: Train a quantum model
    train_parser = subparsers.add_parser("train", help="Train a new quantum model")
    train_parser.add_argument("config", type=str, help="Configuration filename (e.g., density_model.yml)")
    train_parser.add_argument("save_name", type=str, help="Filename to save the trained model")
    train_parser.add_argument("--no-plot", action="store_true", help="Disable training loss plots")

    # COLLECT: Run circuits to generate raw data
    collect_parser = subparsers.add_parser("collect", help="Run circuits on backend to collect raw shots")
    collect_parser.add_argument("hardware", choices=["aer", "ibmq", "ibmqM3"], help="Target backend")
    collect_parser.add_argument("model", type=str, help="Model name")
    collect_parser.add_argument("--points", type=int, default=100, help="Number of input data points")
    collect_parser.add_argument("--shots", type=int, default=100, help="Number of shots (M)")

    # EXPERIMENT: Run scientific benchmarks
    exp_parser = subparsers.add_parser("experiment", help="Run a scientific benchmark experiment")
    exp_parser.add_argument("name", type=str, help="Name of experiment function (e.g., set_size_and_coverage)")
    exp_parser.add_argument("config", type=str, help="Experiment configuration file")
    exp_parser.add_argument("id", type=str, help="Output ID or folder name")

    # PLOT: Generate visualisations
    plot_parser = subparsers.add_parser("plot", help="Generate publication-ready plots")
    plot_parser.add_argument("name", type=str, help="Name of plotting function (e.g., measurement_histograms)")
    plot_parser.add_argument("id", type=str, help="Figure ID associated with the data")

    args = parser.parse_args()

    if args.command == "train":
        from models.trainers import get_trainer 

        trainer = get_trainer(args.config, args.save_name)
        trainer.train(plot_loss=not args.no_plot)
        trainer.save()

    elif args.command == "collect":
        from experiments import data_generation 

        data_generation.run_and_save_jobs(args.hardware, args.model, args.points, args.shots)

    elif args.command == "experiment":
        from experiments import generate_results
    
        if hasattr(generate_results, args.name):
            function = getattr(generate_results, args.name)
            function(args.config, args.id)
        else:
            print(f"Error: Experiment '{args.name}' not found")
            sys.exit(1)

    elif args.command == "plot":
        from visualisation import generate_figures
        
        if hasattr(generate_figures, args.name):
            func = getattr(generate_figures, args.name)
            func(args.id)
        else:
            print(f"Error: Plot '{args.name}' not found")
            sys.exit(1)

if __name__ == "__main__":
    main()

