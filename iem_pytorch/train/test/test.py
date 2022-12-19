import os
import torch as pt
from tqdm import tqdm
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    repo_dir = os.path.dirname(repo_dir)
from kernel import Kernel
from configs.default_config import DefaultConfig


class Test(Kernel):
    def __init__(self,
                 config: DefaultConfig):
        super(Test, self).__init__(config)
        pass

    def calulate_loss(self, vars):
        losses = self.loss(vars)

        losses = {'mse_loss': losses['mse_loss']}
        total_loss = pt.sum(
            pt.stack(
                [pt.mean(v) for k, v in losses.items()],
                dim=0)
        )
        monitors = {}
        with pt.no_grad():
            monitors.update(losses)
        return total_loss, monitors

    def train(self,):
        self.net.train()
        for data in tqdm(self.train_loader, desc='Train', colour='blue'):
            data = {k: v.to(self.device) for k, v in data.items()}
            # print(data)
            self.opt.zero_grad()

            net_outputs = self.net(data)
            net_outputs.update(data)

            loss, monitors = self.calulate_loss(net_outputs)
            loss.backward()
            self.opt.step()

            monitors['lr'] = self.lr_sch.get_last_lr()
            self.writers['train'].train_monitor(monitors)
            self.writers['train'].step()
            self.lr_sch.step()

    @ pt.no_grad()
    def eval(self,):
        self.net.eval()
        for data in tqdm(self.eval_loader, desc='Eval', colour='red'):
            data = {k: v.to(self.device) for k, v in data.items()}
            net_outputs = self.net(data)
            net_outputs.update(data)

            loss, monitors = self.calulate_loss(net_outputs)

            self.writers['eval'].eval_monitor(monitors)
            self.writers['eval'].step()

            net_outputs['x'] = self.normmer.denorm(net_outputs['x'],
                                                   self.normmer.mean[0],
                                                   self.normmer.scale[0])
            net_outputs['pred'] = self.normmer.denorm(net_outputs['pred'],
                                                      self.normmer.mean[1],
                                                      self.normmer.scale[1])
            net_outputs['y'] = self.normmer.denorm(net_outputs['y'],
                                                   self.normmer.mean[1],
                                                   self.normmer.scale[1])
            # print(net_outputs['y'][0], net_outputs['pred'][0])
            self.eval_embs.add_data(attrs=net_outputs)
