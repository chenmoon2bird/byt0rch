import numpy as np
from torch.utils.data import Dataset


class LinearRegression(Dataset):
    def __init__(self,
                 data_path,
                 normmer,
                 is_aug):
        self.normmer = normmer

        self.xs = np.arange(0, 100, dtype=np.float32).reshape(100, 1)
        self.ys = np.arange(0, 200, 2, dtype=np.float32).reshape(100, 1)

        self.xs = self.normmer.norm(self.xs,
                                    self.normmer.mean[0],
                                    self.normmer.scale[0])
        self.ys = self.normmer.norm(self.ys,
                                    self.normmer.mean[1],
                                    self.normmer.scale[1])

        self.xs = self.xs.astype(np.float32)
        self.ys = self.ys.astype(np.float32)

    def __len__(self,):
        return self.xs.shape[0]

    def __getitem__(self, index):
        return self.xs[index], self.ys[index]
