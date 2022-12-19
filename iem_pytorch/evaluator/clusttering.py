import numpy as np
import pandas as pd
import seaborn as sns
import json
from sklearn import manifold
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision.transforms import ToTensor
import os
import sys
repo_dir = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
sys.path.append(repo_dir)
from utils.writer import Writer
from evaluator.evaluator_base import EvaluatorBase


class Clusttering(EvaluatorBase):
    def __init__(self,
                 writer: Writer,
                 embedding_name: str,
                 hue: str,
                 name: str,
                 ):
        self.writer = writer
        self.embedding_name = embedding_name
        self.hue = hue
        self.name = name

    def caculate_tsne(self,
                      df: pd.DataFrame):
        # tic = time.time()
        # embs = [json.loads(x) for x in df[self.embedding_name]]
        embs = df[self.embedding_name].to_numpy()
        embs = np.vstack(embs)
        print('clustering', embs.shape)
        tsne = manifold.TSNE(n_components=2,
                             init='random',
                             learning_rate='auto',
                             n_jobs=-1).fit_transform(embs)
        # random_state=5, verbose=1

        # normalize
        x_min, x_max = tsne.min(0), tsne.max(0)
        tsne = (tsne - x_min) / (x_max - x_min)
        df['tsne_x'] = tsne[:, 0]
        df['tsne_y'] = tsne[:, 1]
        # print('tsne cost:', time.time() - tic)
        return df

    def plot(self,
             df: pd.DataFrame):
        df = self.caculate_tsne(df)
        df = df.sort_values(by=self.hue)
        df[self.hue] = [str(int(x)) for x in df[self.hue]]

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        sns.scatterplot(data=df,
                        x='tsne_x',
                        y='tsne_y',
                        hue=self.hue,
                        ax=ax)
        ax.legend()
        canvas.draw()

        buf = canvas.buffer_rgba()
        img = np.array(buf, dtype=np.uint8)
        img = ToTensor()(img)
        self.writer.add_image(self.name, img, fig)
