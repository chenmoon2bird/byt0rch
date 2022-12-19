import os
import sys
import numpy as np
import torch as pt
from torch.utils.data import DataLoader
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    repo_dir = os.path.dirname(repo_dir)
sys.path.append(repo_dir)
from configs.default_config import DatasetBase
from utils.import_lib import import_model


class DataLoaderBuilder():
    def __init__(self,
                 data_path: str,
                 dataset_conf: DatasetBase,
                 normmer: object):
        self.dataset_conf = dataset_conf

        self.init_dataset(data_path, normmer)
        self.init_data_loader()

    def init_dataset(self,
                     data_path: str,
                     normmer: object):
        args = {'data_path': data_path,
                'normmer': normmer}
        self.dataset = import_model('data_loader', self.dataset_conf, args)

    def init_data_loader(self,):
        self.data_loader = DataLoader(dataset=self.dataset,
                                      batch_size=self.dataset_conf.batch_size,
                                      num_workers=self.dataset_conf.num_workers,
                                      shuffle=self.dataset_conf.shuffle,
                                      collate_fn=self.preprocessing)

    def preprocessing(self, xs):
        transpose_data = list(zip(*xs))
        batchs = {}
        for idx, name in enumerate(self.dataset_conf.outputs):
            batch = np.stack(transpose_data[idx], 0)
            batch = pt.from_numpy(batch)
            batchs[name] = batch
        return batchs
