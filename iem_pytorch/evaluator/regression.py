import numpy as np
import pandas as pd
import seaborn as sns
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


class Regression(EvaluatorBase):
    def __init__(self,
                 writer: Writer,
                 x: str,
                 y: str,
                 name: str,
                 ):
        self.writer = writer
        self.x = x
        self.y = y
        self.name = name

    def plot(self,
             df: pd.DataFrame):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        sns.regplot(data=df,
                    x=self.x,
                    y=self.y,
                    ax=ax)
        ax.legend()
        canvas.draw()
        buf = canvas.buffer_rgba()
        img = np.array(buf, dtype=np.uint8)

        img = ToTensor()(img)
        self.writer.add_image(self.name, img, fig)
