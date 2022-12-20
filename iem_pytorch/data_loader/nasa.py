from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math


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


class NASARandomWalk(Dataset):
    def __init__(self,
                 data_path,
                 sample_columns,
                 sample_len,
                 out_sets,
                 normmer):
        """
        Args:
            data_path: csv path
            normmer: object for normalization
            use_aug: flag for using augmentation
        """
        self.sample_columns = sample_columns
        self.sample_len = sample_len
        self.out_sets = out_sets
        self.normmer = normmer

        self.init_data(data_path)

    def check_norm(self, xs):
        xs_d = xs.shape[-1]
        norm_d = self.normmer.mean_scale.shape[-1]
        if xs_d != norm_d:
            raise BaseException(f'xs_d: {xs_d} not equal norm_d: {norm_d}')

    def init_data(self, data_path):
        df = pd.read_csv(data_path)
        self.col2idx = {col: idx for idx,
                        col in enumerate(self.sample_columns)}

        self.xs = []
        for tg_id, rw in IterDF(df, 'rw_group'):
            outs = [rw[col].to_numpy() for col in self.sample_columns]
            outs = np.stack(outs, axis=1)
            # normalize
            self.check_norm(outs)
            outs = self.normmer.norm(outs, self.normmer.mean_scale)
            self.xs.append(outs)

        self.min_ts = 100000000
        self.max_ts = 0
        for x in self.xs:
            n = len(x)
            if n > self.max_ts:
                self.max_ts = n
            if n < self.min_ts:
                self.min_ts = n
        self.ts_len = self.max_ts + 1
        if self.min_ts < self.sample_len:
            print(f'set sample_len to {self.min_ts}, from {self.sample_len}')
            self.sample_len = self.min_ts

    def __len__(self,):
        return len(self.xs)

    def __getitem__(self, index):
        _xs = self.xs[index]
        [num_t, num_f] = _xs.shape

        xs = np.zeros((self.sample_len, num_f), dtype=np.float32)

        s = np.random.randint(low=0, high=num_t - self.sample_len + 1)
        e = s + self.sample_len

        xs[:self.sample_len] = _xs[s:e]
        outs = []
        for cols in self.out_sets:
            out = [xs[:, self.col2idx[col]] for col in cols]
            out = np.stack(out, axis=1)
            outs.append(out)
        return tuple(outs)
