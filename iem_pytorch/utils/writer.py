import torch as pt
from torch import Tensor
from numpy import ndarray
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict, Union
import mlflow


class Writer():
    def __init__(self, outpath, name, phase):
        self.w = SummaryWriter(outpath)
        self.gstep = 0
        self.gepoch = 0
        self.scalars = {}
        self.cell_reg_df = []
        self.cycle_reg_df = []
        self.name = name
        self.phase = phase

    def step(self,):
        self.gstep += 1

    def epoch(self,):
        self.gepoch += 1

    def train_monitor(self, monitors):
        for n, v in monitors.items():
            self.add_scalar(n, v)

    def eval_monitor(self, monitors):
        for n, v in monitors.items():
            self.add_scalar(n, v, False)

    def add_scalar(self,
                   name,
                   values,
                   write=True):
        """
        Args:
            name: tensor name for showing on tensorboard
            values: tensor size [batch, 1]
            write: write to tensorboard now
        """
        if type(values) != Tensor:
            values = Tensor(values)

        # v_size = values.size()
        # if len(v_size) != 2 and v_size[1] != 1:
        #     raise ValueError(f'{name} size is {v_size} not [batch, 1]')
        values = values.reshape(-1)

        if name not in self.scalars.keys():
            self.scalars[name] = [values]
        else:
            self.scalars[name].append(values)

        if write:
            metric_name = f'{self.name}_Step/{name}'
            value = pt.mean(values)
            self.w.add_scalar(metric_name,
                              value,
                              global_step=self.gstep)
            mlflow.log_metric(key=f'{metric_name}/{self.phase}',
                              value=value.cpu().detach().numpy(),
                              step=self.gstep)

    def add_image(self, name, img, fig):
        metric_name = f'{self.name}_Step/{name}'
        self.w.add_image(metric_name,
                         img,
                         global_step=self.gepoch)
        mlflow.log_figure(figure=fig,
                          artifact_file=f'{metric_name}/{self.phase}.png')

    def summary_scalar(self,):
        for name, values in self.scalars.items():
            metric_name = f'{self.name}_Epoch/{name}'
            value = pt.mean(pt.cat(values, dim=0))
            self.w.add_scalar(metric_name,
                              value,
                              global_step=self.gepoch)
            mlflow.log_metric(key=f'{metric_name}/{self.phase}',
                              value=value.cpu().detach().numpy(),
                              step=self.gepoch)
        for name in self.scalars.keys():
            self.scalars[name] = []

    def close(self,):
        self.w.close()


class WriterGroup():
    def __init__(self,
                 writer_dict: dict,
                 name: str):
        self._writers = {}
        for w_type, infos in writer_dict.items():
            out_path = infos[0]
            phase = infos[1]
            self.writers[w_type] = Writer(out_path, name, phase)

    @property
    def writers(self,) -> Dict[str, Writer]:
        return self._writers

    def step(self,
             phase='all'):
        for w_type, w in self._writers.items():
            if phase == 'all':
                w.step()
            elif w_type == phase:
                w.step()

    def epoch(self,
              phase='all'):
        for w_type, w in self._writers.items():
            if phase == 'all':
                w.epoch()
            elif w_type == phase:
                w.epoch()

    def summary_scalar(self,
                       phase='all'):
        for w_type, w in self._writers.items():
            if phase == 'all':
                w.summary_scalar()
            elif w_type == phase:
                w.summary_scalar()

    def close(self,
              phase='all'):
        for w_type, w in self._writers.items():
            if phase == 'all':
                w.close()
            elif w_type == phase:
                w.close()
