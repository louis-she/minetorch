from __future__ import annotations

import os
from typing import TYPE_CHECKING

import _pickle as pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from minetorch.miner import Miner


class Chart:
    def __init__(self, miner: Miner, name: str):
        self.miner = miner
        self.name = name
        self.init()

    @property
    def code_dir(self):
        return self.miner.code_dir

    def add_points(self, **kwargs):
        """Add points to the chart"""
        raise NotImplementedError()

    def init(self):
        pass


class NoneChart(Chart):
    def init(self):
        pass

    def add_points(self, **kwargs):
        pass


class TensorBoardChart(Chart):
    """Using TensorBoard to visualize charts"""

    def init(self):
        tensorboard_dir = os.path.join(self.code_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

    def add_points(self, **kwargs):
        self.writer.add_scalars(self.name, kwargs, global_step=self.miner.current_epoch)


class ImageFileChart(Chart):
    """Generating charts as local image file"""

    def init(self):
        self.data_file = os.path.join(self.code_dir, f".{self.name}.chart.pkl")
        self.colors = [
            "blue",
            "orange",
            "green",
            "red",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        if os.path.isfile(self.data_file):
            with open(self.data_file, "rb") as f:
                self.chart_data = pickle.load(f)

    def _update_state(self, x, points):
        for key in points:
            if key not in self.chart_data:
                self.chart_data[key] = {}
            self.chart_data[key][x] = points[key]

        # TODO: performance
        with open(self.data_file, "wb") as f:
            pickle.dump(self.state, f)

    def _save_png(self):
        png_file = os.path.join(self.code_dir, self.name + ".png")
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        for index, curve in enumerate(self.chart_data):
            ax.plot(
                *zip(*sorted(self.chart_data[curve].items())),
                label=curve,
                color=self.colors[index],
            )

        ax.legend(loc="upper left")
        fig.savefig(png_file, facecolor="#F0FFFC")
        return png_file

    def scalars(self, **kwargs):
        self._update_state(self.miner.current_epoch, kwargs)
        return self._save_png()
