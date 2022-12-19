import os
import sys
import importlib
from typing import Optional, Union
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    repo_dir = os.path.dirname(repo_dir)
sys.path.append(repo_dir)
from configs.default_config import Layer, BasicModel, DatasetBase, Metric


def import_model(module_name: str,
                 config: Union[Layer, BasicModel, DatasetBase, Metric],
                 args: Optional[dict] = None):

    config_dict = config.dict()
    if args is not None:
        config_dict['args'].update(args)
    name = config_dict['model']
    if 'torch' in name:
        _full_name = name.split('.')
    else:
        _full_name = f'{module_name}.{name}'.split('.')
    _app_name = '.'.join(_full_name[:-1])
    _obj_name = _full_name[-1]
    _app = importlib.import_module(_app_name)
    model = getattr(_app, _obj_name)(**config_dict['args'])
    return model
