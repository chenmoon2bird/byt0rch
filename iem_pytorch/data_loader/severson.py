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


class Severson(Dataset):
    def __init__(self,
                 data_path,
                 sample_columns,
                 out_sets,
                 normmer,
                 use_aug,
                 pos_dim,
                 sample_min: float = 1.):

        self.sample_columns = sample_columns
        self.out_sets = out_sets
        self.normmer = normmer
        self.use_aug = use_aug
        self.pos_dim = pos_dim
        self.sample_min = sample_min

        self.init_data(data_path)

    def check_norm(self, xs):
        xs_d = xs.shape[-1]
        norm_d = self.normmer.mean_scale.shape[-1]
        if xs_d != norm_d:
            raise BaseException(f'xs_d: {xs_d} not equal norm_d: {norm_d}')

    def calculate_time(self,):
        self.min_ts = 100000000
        self.max_ts = 0
        for x in self.xs:
            n = len(x)
            if n > self.max_ts:
                self.max_ts = n
            if n < self.min_ts:
                self.min_ts = n
        self.ts_len = self.max_ts + 1
        if self.sample_min != 1.:
            self.min_ts *= self.sample_min
        print(f'max_len: {self.max_ts}, min_len: {self.min_ts}')

    def init_data(self, data_path):
        df = pd.read_csv(data_path)
        self.col2idx = {col: idx for idx,
                        col in enumerate(self.sample_columns)}

        self.xs = []
        for cycle_id, cycle in IterDF(df, 'cycle_id'):
            outs = [cycle[col].to_numpy() for col in self.sample_columns]
            outs = np.stack(outs, axis=1)

            # normalize
            self.check_norm(outs)
            outs = self.normmer.norm(outs, self.normmer.mean_scale)
            self.xs.append(outs)

        self.calculate_time()

    def __len__(self,):
        return len(self.xs)

    def __getitem__(self, index):
        _xs = self.xs[index]
        # _unix_sec = self.unix_sec[index]
        [num_t, num_f] = _xs.shape

        xs = np.zeros((self.ts_len, num_f), dtype=np.float32)
        using_mask = np.ones((self.ts_len), dtype=np.float32)
        pos = np.zeros((self.ts_len, self.pos_dim), dtype=np.float32)

        if num_t == self.min_ts:
            s = 0
            e = num_t
        elif self.use_aug:
            sample_len = np.random.randint(low=self.min_ts, high=num_t)
            if sample_len == num_t:
                s = 0
                e = num_t
            else:
                s = np.random.randint(low=0, high=num_t - sample_len)
                e = s + sample_len
        else:
            s = 0
            e = num_t
        sample_len = e - s

        # add data
        xs[:sample_len] = _xs[s:e]

        # add using_mask
        using_mask[sample_len:] *= 0

        outs = []
        for cols in self.out_sets:
            out = [xs[:, self.col2idx[col]] for col in cols]
            out.append(using_mask)
            out = np.stack(out, axis=1)
            outs.append(out)
        # outs.append(pos)
        return tuple(outs)
