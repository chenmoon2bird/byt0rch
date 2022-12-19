import torch as pt
from torch import Tensor
import numpy as np
import pandas as pd
import json
from typing import Dict
import os
import sys
repo_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_dir)
from utils.format_transfer import to_np


class EmbeddingContainer():
    def __init__(self,
                 attr_cols: list):
        """
        Args:
            attr_cols
        """
        self.attr_cols = attr_cols
        self.reset()

    def reset(self,):
        self.dfs = []
        self._df = None

    @property
    def df(self,):
        if self._df is None:
            self._df = self.conclusion()
        return self._df

    def add_data(self,
                 attrs: Dict[str, Tensor]):
        """
        Args:
            embs: batch of embedding [batch, embedding dims]
        """
        rows = {n: self.transfer_data(attrs[n]) for n in self.attr_cols}
        # print({n: len(x) for n, x in rows.items()})
        df = pd.DataFrame(rows, columns=self.attr_cols)
        self.dfs.append(df)

    def transfer_data(self, xs):
        if isinstance(xs, pt.Tensor):
            xs = to_np(xs)
            size = xs.shape
            if len(size) == 3:
                xs = np.reshape(xs, (size[0] * size[1], -1))
            size = xs.shape
            if len(size) == 2:
                if size[1] == 1:
                    xs = np.reshape(xs, (-1)).tolist()
                else:
                    xs = [x for x in xs]
        return xs

        # if type(xs) == list:
        #     return [to_np(x) for x in xs if type(x) == pt.Tensor]
        # else:
        #     return to_np(xs)

    def conclusion(self,):
        df = pd.concat(self.dfs, ignore_index=True)
        return df
