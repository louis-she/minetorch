import os
from matplotlib.figure import Figure
from tensorboardX import SummaryWriter
import _pickle as pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class Drawer():
    """To vistualize everything in training process
    """

    def __init__(self, alchemistic_directory, code, state=None):
        """Constructor

        Args:
            alchemistic_directory (string):
                same as trainer's alchemistic_directory
            code (string):
                same as trainer's alchemistic_directory
            graph (state, optional):
                Defaults to None. Since we could draw multiple graph during
                training, the state records the current position of each graph
                The keys are the name of the graphs and values are current
                positions.
        """
        self.step_file = os.path.join(alchemistic_directory, code, '.drawer_step')
        self.code = code
        self.alchemistic_directory = alchemistic_directory

        if state is None:
            self.state = {}
        else:
            self.state = state

    def scalars(self, value, graph):
        """Plot different scalars on a graph

        Args:
            value (dict):
                scalar to plot
            graph (string):
                graph name
        """
        raise NotImplementedError()

    def scalar(self, value, graph):
        """Plot one scalar on a graph

        Args:
            value (float):
                scalar to plot
            graph (string):
                graph name
        """
        self.scalars({graph: value}, graph)

    def get_state(self):
        """Return current state(counter) of the Drawer
        """
        return self.state

    def set_state(self, state):
        """Set current state(counter) to state
        """
        self.state = state


class TensorboardDrawer(Drawer):
    """To vistualize everything in training process using tensorboard
    """

    def __init__(self, alchemistic_directory, code, step=None):
        super().__init__(alchemistic_directory, code, step)
        self.writer = SummaryWriter(log_dir=os.path.join(
            alchemistic_directory, code
        ))

    def scalars(self, value, graph):
        """Add a scalar on a graph

        Args:
            value (dict):
                scalars to put on the graph
            graph (string):
                graph name
        """
        if graph not in self.state:
            self.state[graph] = 0
        key = '{}/{}'.format(self.code, graph)
        if isinstance(value, dict):
            self.writer.add_scalars(key, value, self.state[graph])
        else:
            self.writer.add_scalar(key, value, self.state[graph])
        self.state[graph] += 1


class MatplotlibDrawer(Drawer):

    def __init__(self, alchemistic_directory, code, state=None):
        super().__init__(alchemistic_directory, code, state)
        self.graph_dir = os.path.join(alchemistic_directory, code, 'graphs')
        self.data_file = os.path.join(self.graph_dir, '.graphs.pickle')
        self.colors = ['blue', 'orange', 'green', 'red', 'purple',
                       'brown', 'pink', 'gray', 'olive', 'cyan']
        if not os.path.isdir(self.graph_dir):
            os.mkdir(self.graph_dir)
        if os.path.isfile(self.data_file):
            with open(self.data_file, 'rb') as f:
                self.graph_data = pickle.load(f)

    def _update_state(self, values, graph):
        if graph not in self.state or not isinstance(self.state[graph], dict):
            self.state[graph] = {}
        for key in values:
            if key not in self.state[graph]:
                self.state[graph][key] = []
            self.state[graph][key].append(values[key])
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.state, f)

    def _save_png(self, graph):
        png_file = os.path.join(self.graph_dir, graph + ".png")
        fig = Figure()
        FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(True)
        for index, curve in enumerate(self.state[graph]):
            ax.plot(self.state[graph][curve], label=curve, color=self.colors[index])

        ax.legend(loc='upper left')
        fig.savefig(png_file)

    def scalars(self, values, graph):
        """Add a scalar on a graph

        Args:
            value (dict):
                scalars to put on the graph
            graph (string):
                graph name
        """
        self._update_state(values, graph)
        self._save_png(graph)
