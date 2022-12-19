from pickletools import optimize
from random import shuffle
from typing import Optional, List, Dict
from pydantic import BaseModel
import yaml
import os
repo_dir = os.path.dirname(os.path.realpath(__file__))


class TrainingSettings(BaseModel):
    kernel: str
    mode: str = 'train_eval'
    device: str = '0'
    max_epoch: int
    pretrained_model: Optional[str] = None


class BasicModel(BaseModel):
    model: str
    args: dict


class Layer(BaseModel):
    name: str
    model: str
    args: dict
    inputs: List[str]
    outputs: List[str]


class Model(BaseModel):
    inputs: List[str]
    outputs: List[str]
    layers: List[Layer]


class Loss(BaseModel):
    layers: List[Layer]


class DatasetBase(BaseModel):
    batch_size: Optional[int] = 2
    shuffle: bool = False
    num_workers: int = 4
    model: str
    args: dict
    outputs: List[str]


class Data(BaseModel):
    training: str
    testing: str
    normalizer: BasicModel
    dataset: Dict[str, DatasetBase]
    # dataloader: dict


class Optimizer(BaseModel):
    optimizer: BasicModel
    lr_scheduler: BasicModel


class Metric(BaseModel):
    name: str
    writer: str
    model: str
    args: dict


class Evaluator(BaseModel):
    include: List[str]
    metrics: List[Metric]


class DefaultConfig(BaseModel):
    mkflow_uri: str = "http://localhost:5000"
    experiment_name: str = "IEM"
    run_name: str
    experiment_dir: str
    params: dict
    run_dir: str = 'run_default'
    training_settings: TrainingSettings
    data: Data
    optimizer: Optimizer
    model: Model
    loss: Loss
    evaluator: Evaluator


if __name__ == '__main__':
    from pprint import pprint
    config = {
        'project_name': 'qq',
        'experiment_dir': 'qq',
        'training_settings': {
            'device': 0,
            'max_epoch': 200,
            'batch_size': 3,
            'pretrained_model': 'qq',
        },
        'data': {
            'num_workers': 4,
            'training': 'qq',
            'testing': 'qq',
            'normalizer': {
                'model': 'nasa.NASANorm',
                'args': {
                    'mean_std_file': 'qq',
                    'data_folder': 'qq',
                }
            }
        },
        'model': {
            'inputs': ['x', 'y'],
            'outputs': ['repres'],
            'layers': [
                {'name': 'encoder',
                 'model': 'nasa_rw.NASATransAE',
                 'inputs': ['x'],
                 'outputs': ['rul']}
            ]
        },
        'optimizer': {
            'optimizer': {
                'model': 'torch.optim.AdamW',
                'args': {
                    'lr': 1e-5,
                    'weight_decay': 5e-3,
                    'amsgrad': True
                }
            },
            'lr_scheduler': {
                'model': 'cosine.CosineAnnealingWarmupRestarts',
                'args': {
                    'max_lr': 1e-3,
                    'min_lr': 1e-5,
                }
            }
        },
        'loss_function': {


        },
    }

    config = DefaultConfig(**config)
    pprint(config.dict())
    file_dir = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(file_dir, 'default_config.yaml')
    with open(yaml_file, 'w') as f:
        yaml.safe_dump(config.dict(), f, indent=4)
