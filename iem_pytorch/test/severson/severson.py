import torch as pt
import pandas as pd
import numpy as np
import math
import yaml
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(2):
    repo_dir = os.path.dirname(repo_dir)
sys.path.append(repo_dir)
from utils import config_parser
from utils.import_lib import import_model


class IterDF:
    def __init__(self, df, col):
        self.df = df
        self.col = col
        self.index = 0
        self.ids = self.df[self.col].unique().tolist()
        self.max_num = len(self.ids)

    def __iter__(self,):
        return self

    def __next__(self,):
        if self.index < self.max_num:
            id = self.ids[self.index]
            _df = self.df[self.df[self.col] == id]
            self.index += 1
            return id, _df
        else:
            raise StopIteration


args = config_parser.get_args()
config = config_parser.process_args(args)
device = f'cuda:{config.training_settings.device}'

model_path = f'{config.run_dir}/model.pt'
net = pt.load(model_path).to(device)
net.eval()

data_conf = config.data
normmer_args = {'norm_path': f'{config.run_dir}'}
normmer = import_model('normalization',
                       data_conf.normalizer,
                       normmer_args)

csv_path = f'{config.data.training}'
df = pd.read_csv(csv_path)
cols = config.data.dataset['train'].args['sample_columns']
col2idx = {col: idx for idx, col in enumerate(cols)}


def get_xs_from_tg(df, cols):
    xs = [df[col].to_numpy() for col in cols]
    xs = np.stack(xs, axis=1)
    # normalize
    xs = normmer.norm(xs, normmer.mean_scale)
    return xs


def get_xs(style, cols, trip):
    out_sets = config.data.dataset['train'].args['out_sets']
    output_names = config.data.dataset['train'].outputs

    # xs = get_xs_from_tg(tg, cols)
    xs = get_xs_from_tg(trip, cols)

    # tgs = get_xs_from_tg(tg0, cols)
    tgs = get_xs_from_tg(style, cols)

    outs = {}
    for idx, cols in enumerate(out_sets):
        out_name = output_names[idx]
        if out_name == 'xs_sty':
            use_mask = np.ones_like(tgs[:, 0])
            out = [tgs[:, col2idx[col]] for col in cols]
            out.append(use_mask)
            out = np.stack(out, axis=1)
        else:
            use_mask = np.ones_like(xs[:, 0])
            out = [xs[:, col2idx[col]] for col in cols]
            out.append(use_mask)
            out = np.stack(out, axis=1)
        out = pt.from_numpy(out).to(device).float()
        out = out.unsqueeze(dim=0)
        outs[out_name] = out
    return outs


tgs = df.cycle_id.to_numpy()
tg0 = df[df.cycle_id == tgs[0]]
emb_df = []
outs = []
for tg_id, tg in IterDF(df, 'cycle_id'):
    style = tg.copy()
    trip = tg0.copy()
    xs = get_xs(style=style,
                cols=cols,
                trip=trip)
    # x = pt.from_numpy(x).to(device).float()
    # x = x.unsqueeze(dim=0)
    # print(x.size())

    net_outs = net(xs)
    y = net_outs['sty_preds'].cpu().detach().numpy()[:, :, 0:1]
    y = normmer.denorm(y, normmer.mean_scale[:, 0])[0, :, 0]
    # print(y.shape)
    _tg = trip.copy()
    _tg['volt_fix'] = y

    # sty_mean = net_outs['sty_mean'].cpu().detach().numpy()[0, 0, :]
    # feats = net_outs['feature'].cpu().detach().numpy()[0, 0, :]
    # dist = np.mean(np.power(sty_mean - feats, 2))
    # print(dist)

    y = net_outs['sty_preds'].cpu().detach().numpy()[:, :, 1:2]
    y = normmer.denorm(y, normmer.mean_scale[:, 1])[0, :, 0]
    _tg['amp_fix'] = y

    _tg['cycle_id'] = [tg_id for _ in range(len(_tg))]
    outs.append(_tg)

    # trip_embs = net_outs['feature'].cpu().detach().numpy()[:, 0]
    # sty_embs = net_outs['sty_mean'].cpu().detach().numpy()[0]
    # dist = np.sum(np.power(trip_embs - sty_embs, 2))
    # # dist = cosine_similarity(trip_embs, sty_embs)[0, 0]
    # emb = [tg_id, dist]
    # emb_df.append(emb)


outs = pd.concat(outs, ignore_index=True)
out_path = f'{config.run_dir}/out_data_trans_new.csv'
outs.to_csv(out_path, index=False)
