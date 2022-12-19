from utils import config_parser
import importlib
from configs.default_config import DefaultConfig
import torch
import random
import numpy as np


def get_kernel(config: DefaultConfig):
    kernel_name = f'train.{config.training_settings.kernel}'.split('.')
    _app_name = '.'.join(kernel_name[:-1])
    _obj_name = kernel_name[-1]
    _app = importlib.import_module(_app_name)
    return getattr(_app, _obj_name)(config)


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    args = config_parser.get_args()
    config = config_parser.process_args(args)

    pytorch_framework = get_kernel(config=config)
    pytorch_framework.run()
