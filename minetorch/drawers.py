import os

from tensorboardX import SummaryWriter


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
            step (int, optional):
                Defaults to None. The timeline
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
