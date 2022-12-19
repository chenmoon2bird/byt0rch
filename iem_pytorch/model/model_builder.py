import os
import sys
from typing import Dict, List
from torch import nn, Tensor
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    repo_dir = os.path.dirname(repo_dir)
sys.path.append(repo_dir)
from configs.default_config import Model
from utils.import_lib import import_model


class Nets(nn.Module):
    def __init__(self,
                 model_infos: Model):
        super(Nets, self).__init__()
        self.model_infos = model_infos
        self.models = []
        self.vars = {}
        for idx, infos in enumerate(self.model_infos.layers):
            model = import_model('model', infos)
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

    def get_inputs(self,
                   idx) -> List[Tensor]:
        input_names = self.model_infos.layers[idx].inputs
        return [self.vars[x] for x in input_names]

    def fetch_outputs(self,
                      idx,
                      outputs):
        output_names = self.model_infos.layers[idx].outputs

        if isinstance(outputs, tuple):
            for idx, output_name in enumerate(output_names):
                self.vars[output_name] = outputs[idx]
        else:
            self.vars[output_names[0]] = outputs

    def forward(self, *args):
        for idx, name in enumerate(self.model_infos.inputs):
            self.vars[name] = args[idx]

        for idx, model in enumerate(self.models):
            inputs = self.get_inputs(idx)
            outputs = model(*inputs)
            self.fetch_outputs(idx, outputs)

        outs = tuple([self.vars[name] for name in self.model_infos.outputs])
        # del self.vars
        # self.vars = {}
        return outs


class ModelBuilder():
    def __init__(self,
                 model_infos: Model,
                 device):
        self.model_infos = model_infos
        self.nets = Nets(self.model_infos).to(device)

    def run(self,
            x: Dict[str, Tensor]) -> Dict[str, Tensor]:

        xs = [x[name] for name in self.model_infos.inputs]

        outs = self.nets(*xs)

        outs = {name: outs[idx]
                for idx, name in enumerate(self.model_infos.outputs)}
        return outs
