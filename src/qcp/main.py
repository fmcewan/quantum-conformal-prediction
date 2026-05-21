# Standard library imports
import argparse

from qcp.prediction.conformal_predictor import ConformalPredictor
from qcp.utilities.file_handling import load_yaml
from qcp.models.factory import get_trainer 
from qcp.collection import shot_collector 

def main():

    # Initialise the main parser
    parser = argparse.ArgumentParser(description="Pipeline Manager")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available execution modes")

    # TRAIN: Train a quantum model
    train_parser = subparsers.add_parser("train", help="Train a new quantum model")
    train_parser.add_argument("config", type=str, help="Specification name (e.g. standard_normal)")
    train_parser.add_argument("save_name", type=str, help="Filename to save the trained model")
    train_parser.add_argument("--no-plot", action="store_true", help="Disable training loss plots")

    # COLLECT: Run circuits to generate raw data
    collect_parser = subparsers.add_parser("collect", help="Run circuits on backend to collect raw shots")
    collect_parser.add_argument("hardware", choices=["aer", "ibmq", "ibmqM3"], help="Target backend")
    collect_parser.add_argument("model", type=str, help="Model name")
    collect_parser.add_argument("--points", type=int, default=100, help="Number of input data points")
    collect_parser.add_argument("--shots", type=int, default=100, help="Number of shots (M)")

    # PREDICT: Perform conformal prediction 
    predict_parser = subparsers.add_parser("predict", help="Run conformal prediction and save raw results")
    predict_parser.add_argument("protocol", type=str, help="Protocol name (e.g. predictor_evaluation)")

    args = parser.parse_args()

    if args.command == "train":
        trainer = get_trainer(args.config, args.save_name)
        
        trainer.train()
        trainer.save()

    elif args.command == "collect":
        shot_collector.run_and_save_jobs(args.hardware, args.model, args.points, args.shots)
    
    elif args.command == "predict":
        protocol_path = f"protocols/{args.protocol}.yml"
        configuration = load_yaml(protocol_path)
        output_directory = f"data/results/{args.protocol}"
        
        excluded_keys = {'name', 'algorithms'}
        common_properties = {k: v for k, v in configuration.items() if k not in excluded_keys}
        for algorithm in configuration['algorithms']:
            current_configuration = common_properties | algorithm
            conformal_predictor = ConformalPredictor(current_configuration)
            conformal_predictor.run(output_directory, algorithm['name']) 

if __name__ == "__main__":
    main()

