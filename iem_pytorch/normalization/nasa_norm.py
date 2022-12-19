import os
import numpy as np
import pandas as pd
repo_dir = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):
    repo_dir = os.path.dirname(repo_dir)


class NASANorm():
    def __init__(self,
                 norm_path,
                 data_path,
                 norm_cols):
        self.norm_cols = norm_cols
        norm_path = f'{norm_path}/norm.npy'
        if os.path.exists(norm_path):
            print(f'Load norm from: {norm_path}')
            self.mean_scale = np.load(norm_path)
        else:
            self.mean_scale = self.get_mean_scale(data_path)
            np.save(norm_path, self.mean_scale)
            print(f'Save norm to: {norm_path}')

    def get_mean_scale(self, data_path):
        def to_np(x):
            return np.array(x, dtype=np.float32)
        df = pd.read_csv(data_path)
        mean_scale = []
        for name_method in self.norm_cols:
            column = name_method['column']
            method = name_method['method']
            vs = df[column].to_numpy()
            if isinstance(method, list):
                mean_scale.append(to_np(method))
            elif method == 'std_norm':
                mean = np.mean(vs)
                scale = np.std(vs)
                mean_scale.append(to_np([mean, scale]))
            elif method == 'zero_abs_max':
                mean = 0
                scale = np.max(np.abs(vs))
                mean_scale.append(to_np([mean, scale]))
            elif method == 'zero_abs_mean':
                mean = 0
                scale = np.mean(np.abs(vs))
                mean_scale.append(to_np([mean, scale]))
        mean_scale = np.stack(mean_scale, axis=1)
        return mean_scale

    def norm(self, x, mean_scale):
        mean = mean_scale[0]
        scale = mean_scale[1]
        return (x - mean) / scale

    def denorm(self, x, mean_scale):
        mean = mean_scale[0]
        scale = mean_scale[1]
        return x * scale + mean
