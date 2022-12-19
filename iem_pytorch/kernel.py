import os
from time import time
from typing import Any, Callable
import torch as pt
from torchinfo import summary
import mlflow


from configs.default_config import DefaultConfig
from data_loader.data_loader_builder import DataLoaderBuilder
from model.model_builder import ModelBuilder
from loss.loss_builder import LossBuilder
from utils.writer import WriterGroup
from utils.storage import EmbeddingContainer
from utils.config_parser import save_config
from utils.import_lib import import_model


def train_unimplemented(self, epoch):
    raise NotImplementedError(f'Train is missing the required')


def eval_unimplemented(self, epoch):
    raise NotImplementedError(f'Eval is missing the required')


def plot_metrics_unimplemented(self,):
    raise NotImplementedError(f'plot_metrics is missing the required')


class Kernel():
    def __init__(self,
                 config: DefaultConfig):
        self.config = config

        self.init_output_dir()
        self.init_mlflow()
        self.setting_mode()

        self.init_device()
        self.init_data_loader()
        self.init_model()
        self.init_opt()
        self.init_loss()
        self.init_writer()
        # self._train = None
        # self._eval = None
    train = train_unimplemented
    eval = eval_unimplemented
    plot_metrics = plot_metrics_unimplemented

    def init_output_dir(self,):
        os.makedirs(self.config.run_dir,
                    exist_ok=True)
        save_config(self.config, f'{self.config.run_dir}/config.yaml')
        print(f'save run to: {self.config.run_dir}')

    def init_mlflow(self,):
        # check experiment exsists
        mlflow.set_tracking_uri(self.config.mkflow_uri)
        print(f'MLFLOW from uri: {self.config.mkflow_uri}')
        exp = mlflow.get_experiment_by_name(self.config.experiment_name)
        if exp is None:
            self.exp_id = mlflow.create_experiment(self.config.experiment_name)
            print(f'MLFLOW create experiment id: {self.exp_id}')
        else:
            self.exp_id = exp.experiment_id
            print(f'MLFLOW experiment id: {self.exp_id}')

    def setting_mode(self,):
        mode = self.config.training_settings.mode
        if mode == 'train' or mode == 'train_eval':
            self.only_train = True
        else:
            self.only_train = False

    def init_device(self,):
        if pt.cuda.is_available():
            device = f'cuda:{self.config.training_settings.device}'
        else:
            device = 'cpu'
        self.device = pt.device(device)
        print('device:', self.device)

    def init_data_loader(self,):
        data_conf = self.config.data

        # init normalizer
        normmer_args = {'norm_path': f'{self.config.run_dir}'}
        self.normmer = import_model('normalization',
                                    data_conf.normalizer,
                                    normmer_args)

        # init train loader
        self.train_loader = DataLoaderBuilder(data_path=data_conf.training,
                                              dataset_conf=data_conf.dataset['train'],
                                              normmer=self.normmer).data_loader

        # init eval loader
        self.eval_loader = DataLoaderBuilder(data_path=data_conf.testing,
                                             dataset_conf=data_conf.dataset['eval'],
                                             normmer=self.normmer).data_loader

    def init_model(self,):
        model_conf = self.config.model
        self.net = ModelBuilder(model_conf, self.device)
        summary(self.net.nets)

    def init_opt(self,):
        opt_conf = self.config.optimizer.optimizer
        opt_args = {'params': self.net.nets.parameters()}
        self.opt = import_model('opimizer', opt_conf, opt_args)

        tl_len = len(self.train_loader)
        max_epoch = self.config.training_settings.max_epoch
        lr_sch_conf = self.config.optimizer.lr_scheduler
        lr_sch_args = {'optimizer': self.opt,
                       'first_cycle_steps': max_epoch * tl_len,
                       'warmup_steps': 5 * tl_len}
        self.lr_sch = import_model('lr_scheduler', lr_sch_conf, lr_sch_args)

    def init_loss(self,):
        loss_conf = self.config.loss
        self.loss = LossBuilder(loss_conf).to(self.device)

    def init_writer(self,):
        tb_path = f'{self.config.run_dir}/tensorboard'
        print(f'save tensorboard to {tb_path}')
        writer_names = ['train', 'eval']
        writer_dict = {n: [os.path.join(tb_path, n), n] for n in writer_names}
        self.writer_group = WriterGroup(writer_dict=writer_dict,
                                        name='EOL')

        self.writers = self.writer_group.writers

        attr_cols = self.config.evaluator.include
        self.eval_embs = EmbeddingContainer(attr_cols=attr_cols)

        self.metrics = {}
        metric_confs = self.config.evaluator.metrics
        for metric_conf in metric_confs:
            name = metric_conf.name
            args = {'name': name,
                    'writer': self.writers[metric_conf.writer]}
            self.metrics[name] = import_model('evaluator', metric_conf, args)

    def train_step(self, monitors):
        self.writers['train'].train_monitor(monitors)
        self.writers['train'].step()
        self.lr_sch.step()

    def eval_step(self, monitors):
        self.writers['eval'].eval_monitor(monitors)
        self.writers['eval'].step()

    # def plot_metrics(self,):
    #     df = self.eval_embs.conclusion()
    #     for n, f in self.metrics.items():
    #         f.plot(df)
    #     self.eval_embs.reset()
    def save_model(self,):
        out_path = f'{self.config.run_dir}/model.pt'
        pt.save(self.net.nets, out_path)
        # mlflow.pytorch.log_model(self.net.nets, 'model')
        # self.save_torchserve()
        # self.save_scirpt()

    def save_scirpt(self,):
        inputs = []
        for data in self.train_loader:
            data = {k: v.to(self.device) for k, v in data.items()}
            inputs = [data[n] for n in self.config.model.inputs]
            break
        scripted_pytorch_model = pt.jit.script(self.net.nets)
        mlflow.pytorch.log_model(pytorch_model=scripted_pytorch_model,
                                 artifact_path="scripted_model",
                                 code_paths=os.path.realpath(__file__))

    # def save_torchserve(self,):
    #     # check inputs size
    #     # model = pt.load(f'{self.config.run_dir}/model.pt')
    #     inputs = []
    #     for data in self.train_loader:
    #         data = {k: v.to(self.device) for k, v in data.items()}
    #         inputs = [data[n] for n in self.config.model.inputs]
    #         break
    #     # model.eval()
    #     model = pt.jit.trace(self.net.nets, tuple(inputs))
    #     out_path = f'{self.config.run_dir}/model_ts.pt'
    #     model.save(out_path)

    def run(self,):
        self.max_epoch = self.config.training_settings.max_epoch
        with mlflow.start_run(experiment_id=self.exp_id,
                              run_name=self.config.run_name):
            # log artifact
            mlflow.log_artifact(f'{self.config.run_dir}/config.yaml')
            mlflow.log_params(self.config.params)
            for epoch in range(self.max_epoch):
                tic = time()
                print(f'\nepoch: {epoch}/{self.max_epoch}')
                self.train(epoch)
                self.eval(epoch)

                self.writer_group.summary_scalar()

                self.writer_group.epoch()
                print('epoch cost:', time() - tic)
                if epoch == (self.max_epoch - 1):
                    # if epoch == 0:
                    self.plot_metrics()
                self.eval_embs.reset()
            self.writer_group.close()
            self.save_model()
