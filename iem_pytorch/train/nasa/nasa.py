import os
import torch as pt
from tqdm import tqdm
import pandas as pd
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    repo_dir = os.path.dirname(repo_dir)
from kernel import Kernel
from configs.default_config import DefaultConfig


class ZemoTrainer(Kernel):
    def __init__(self,
                 config: DefaultConfig):
        super(ZemoTrainer, self).__init__(config)
        pass

    def calulate_loss(self, vars):
        losses = self.loss(vars)

        # train_losses = {
        #     'mse_loss': losses['mse_loss']
        # }
        # total_loss = pt.sum(
        #     pt.stack(
        #         [pt.mean(v) for k, v in train_losses.items()],
        #         dim=0)
        # )
        # with pt.no_grad():
        #     monitors = {
        #         'mse_loss_volt': losses['mse_loss_volt'],
        #         'mse_loss_use_mask': losses['mse_loss_use_mask']
        #     }
        #     monitors.update(train_losses)
        train_losses = {n: loss for n, loss in losses.items() if 'loss' in n}
        total_loss = pt.sum(
            pt.stack(
                [pt.mean(v) for k, v in train_losses.items()],
                dim=0)
        )
        with pt.no_grad():
            monitors = {n: loss for n, loss in losses.items()
                        if 'monitor' in n}
            monitors.update(train_losses)
        return total_loss, monitors

    def plot_metrics(self,):
        df = self.eval_embs.conclusion()
        # print(df.t_use_mask.unique())
        for n, f in self.metrics.items():
            if n == 'rc_feats':
                _df = []
                if 'time_group' in df.columns:
                    col = 'time_group'
                else:
                    col = 'rw_group'
                for tg_id in df[col].unique():
                    tg = df[df[col] == tg_id]
                    _df.append(tg.sample(n=1))
                _df = pd.concat(_df, ignore_index=True)
            elif n == 'volt':
                if 't_use_mask' in df.columns:
                    _df = df[df.t_use_mask > 0]
                    _df = _df.sample(frac=0.1, replace=False)
                else:
                    _df = df.sample(frac=0.1, replace=False)
            else:
                _df = df
                _df = _df.sample(frac=0.1, replace=False)
            f.plot(_df)

    def train(self, epoch):
        self.net.nets.train()
        for data in tqdm(self.train_loader, desc='Train', colour='blue'):
            data = {k: v.to(self.device) for k, v in data.items()}
            # print(data)
            self.opt.zero_grad()

            net_outputs = self.net.run(data)
            net_outputs.update(data)

            loss, monitors = self.calulate_loss(net_outputs)
            loss.backward()
            self.opt.step()

            monitors['lr'] = self.lr_sch.get_last_lr()
            self.train_step(monitors)

    @ pt.no_grad()
    def eval(self, epoch):
        self.net.nets.eval()
        for data in tqdm(self.eval_loader, desc='Eval', colour='red'):
            data = {k: v.to(self.device) for k, v in data.items()}
            net_outputs = self.net.run(data)
            net_outputs.update(data)

            loss, monitors = self.calulate_loss(net_outputs)

            self.eval_step(monitors)
            # net_outputs['t_volt_denorm'] = self.normmer.denorm(net_outputs['t_volt'],
            #                                                    self.normmer.mean_scale[:, -1])
            # net_outputs['volt_denorm'] = self.normmer.denorm(net_outputs['t_volt'],
            #                                                    self.normmer.mean_scale[:, -1])
            # print(net_outputs['y'][0], net_outputs['pred'][0])
            if epoch == (self.max_epoch - 1):
                self.eval_embs.add_data(attrs=net_outputs)
