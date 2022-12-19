import os
import yaml
import argparse
import easydict
repo_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
from configs.default_config import DefaultConfig


def get_args():
    """get args from bash command line
    """
    default_config = 'configs/default_config.yaml'

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config',
                        type=str,
                        default=default_config,
                        help='Configuration file (*.yaml) path')
    parser.add_argument('-d', '--device',
                        type=str,
                        default='0',
                        help='GPU device for training (e.g. 0,1,2,3 or 0-7)')
    parser.add_argument('-m', '--mode',
                        type=str,
                        default='train_eval',
                        help='model: train, eval, train_eval')

    args = parser.parse_args()
    return args


def process_args(args) -> DefaultConfig:
    """wrapper func
    """
    config = process_config(args.config)
    config.training_settings.device = process_device(args.device)
    config.training_settings.mode = args.mode
    config.run_dir = f'{config.experiment_dir}/{config.experiment_name}/{config.run_name}'
    return config


def process_config(config_file) -> DefaultConfig:
    """merge config between `default_config` and `your_config`
    """
    config = get_config_from_yaml(config_file)
    return config


def save_config(config: DefaultConfig, path: str):
    with open(path, 'w') as config_file:
        yaml.safe_dump(config.dict(),
                       config_file,
                       sort_keys=False)


def get_config_from_yaml(yaml_file) -> DefaultConfig:
    """
    Get the config from a json file
    :param yaml_file:
    :return: config(namespace) or config(dictionary)
    """
    # Parse the configurations from the config yaml file provided
    with open(yaml_file, 'r') as config_file:
        config_dict = yaml.safe_load(config_file)
    config = DefaultConfig(**config_dict)
    return config


def process_device(gpu_ids):
    if '-' in gpu_ids:
        gpus = gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        processed_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
        return processed_ids
    else:
        return gpu_ids
