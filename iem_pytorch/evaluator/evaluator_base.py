import pandas as pd


def plot_unimplemented(self, df: pd.DataFrame):
    raise NotImplementedError(f'plot is missing the required')


class EvaluatorBase():
    def __init__(self,):
        self._plot = None

    plot = plot_unimplemented
