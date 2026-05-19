from qcp.utilities.file_handling import load_yaml 

from qcp.models.trainers.unsupervised_trainer import UnsupervisedTrainer

def get_trainer(configuration, save_name):

    configuration_data = load_yaml(f"specifications/{configuration}")
    trainer_type = configuration_data['training']['trainer']
    
    if trainer_type == "unsupervised":
        return UnsupervisedTrainer(configuration_data, save_name)
    elif trainer_type == "supervised":
        from qcp.models.trainers.regression import RegressionTrainer
        return RegressionTrainer(configuration_data, save_name)
    elif trainer_type == "classification":
        from qcp.models.trainers.classification import ClassificationTrainer
        return ClassificationTrainer(config_data, save_name)
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")
