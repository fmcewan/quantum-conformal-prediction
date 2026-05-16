import yaml
import os
import math

def save_config_in_figure_folder(figure_id, config, config_name='config.yaml'):
    """
    saves a config file in a figure folder
    """

    save_directory = f'saved/figures/figure_{figure_id}/'
    os.makedirs(save_directory, exist_ok=True)
    config_dst = os.path.join(save_directory, config_name)

    if isinstance(config, str):
        with open("configurations/results/" + config, 'r') as file:
            config = yaml.safe_load(file)
    
    with open(config_dst, 'w') as file:
        yaml.dump(config, file)
        
    print(f"Config file saved at: {config_dst}")

def create_log_spaced_points(dictionary):

    start = dictionary['start']
    end = dictionary['end']
    resolution = dictionary['resolution']
    ratio = (end/start)**(1/(resolution-1))
    return [math.ceil(start*(ratio**i)) for i in range(resolution)]
