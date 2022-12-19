import os
import sys
from typing import List
from torch import nn, Tensor
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    repo_dir = os.path.dirname(repo_dir)
sys.path.append(repo_dir)
from configs.default_config import Loss
from utils.import_lib import import_model


class LossBuilder(nn.Module):
    """Some Information about LossBuilder"""

    def __init__(self,
                 loss_infos: Loss):
        super(LossBuilder, self).__init__()
        self.loss_infos = loss_infos
        self.losses = []
        self.vars = {}
        self.output_names = []

        for idx, infos in enumerate(self.loss_infos.layers):
            self.output_names.extend(infos.outputs)
            loss = import_model('loss', infos)
            self.losses.append(loss)
        self.losses = nn.ModuleList(self.losses)

    def get_inputs(self,
                   idx) -> List[Tensor]:
        input_names = self.loss_infos.layers[idx].inputs
        return [self.vars[x] for x in input_names]

    def fetch_outputs(self,
                      idx,
                      outputs):
        output_names = self.loss_infos.layers[idx].outputs

        if isinstance(outputs, tuple):
            for idx, output_name in enumerate(output_names):
                self.vars[output_name] = outputs[idx]
        else:
            self.vars[output_names[0]] = outputs

    def check_output(self, name):
        value = self.vars[name]
        v_size = value.size()
        if len(v_size) != 2 and v_size[1] != 1:
            raise ValueError(f'{name} size is {v_size} not [batch, 1]')
        return value

    def forward(self, xs):
        # print(xs)
        for info in self.loss_infos.layers:
            self.vars.update({n: xs[n] for n in info.inputs})

        for idx, model in enumerate(self.losses):
            inputs = self.get_inputs(idx)
            outputs = model(*inputs)

            self.fetch_outputs(idx, outputs)

        outs = {name: self.check_output(name) for name in self.output_names}
        return outs
