import peewee

import minetorch.constants as C
from minetorch.logger import get_runtime_logger
from minetorch.orm import Experiment, Graph, Timer

from minetorch.proto import minetorch_pb2, minetorch_pb2_grpc


class MinetorchServicer(minetorch_pb2_grpc.MinetorchServicer):

    # TODO: what about experiment changed?
    caches = {}

    timer_mapping = {
        C.TIMER_ITERATION: 1,
        C.TIMER_EPOCH: 2,
        C.TIMER_SNAPSHOT: 3,
    }

    def _get_experiment(self, experiment_id):
        if experiment_id in self.caches:
            return self.caches[experiment_id], None
        try:
            experiment = Experiment.get_by_id(experiment_id)
            self.caches[experiment_id] = experiment
        except peewee.DoesNotExist:
            return False, minetorch_pb2.StandardResponse(
                status=1,
                message='Could not find experiment, abort'
            )
        return experiment, None

    def CreateGraph(self, request, context):
        # TODO: performance
        experiment, err = self._get_experiment(request.experiment_id)
        if not experiment:
            return err
        try:
            timer = experiment.timers.where(Timer.category == self.timer_mapping[request.timer_category]).get()
        except peewee.DoesNotExist:
            # TODO: here we should notify the user to create the timer first!
            return minetorch_pb2.StandardResponse(
                status=100,
                message='Timer does not exists, please create the correct timer before create graph'
            )
        try:
            Graph.create(
                experiment_id=experiment.id,
                snapshot_id=experiment.current_snapshot().id,
                name=request.graph_name,
                timer=timer
            )
        except peewee.IntegrityError:
            pass
        return minetorch_pb2.StandardResponse(
            status=0,
            message='ok'
        )

    def AddPoint(self, request, context):
        # TODO: performance
        experiment, err = self._get_experiment(request.experiment_id)
        if not experiment:
            return err
        graph = experiment.graphs.where(Graph.name == request.graph_name).get()
        graph.add_point(graph.timer.current, request.y)
        return minetorch_pb2.StandardResponse(
            status=0,
            message='ok'
        )

    def SetTimer(self, request, context):
        # TODO: performance
        experiment, err = self._get_experiment(request.experiment_id)
        if not experiment:
            return err
        try:
            timer = experiment.timers.where(Timer.category == self.timer_mapping[request.category]).get()
            timer.current = request.current
            timer.save()
        except peewee.DoesNotExist:
            timer = Timer.create(
                experiment_id=experiment.id,
                snapshot_id=experiment.current_snapshot().id,
                category=self.timer_mapping[request.category],
                current=request.current
            )
        return minetorch_pb2.StandardResponse(
            status=0,
            message='ok'
        )

    def SendLog(self, request, context):
        try:
            experiment = Experiment.get_by_id(request.experiment_id)
        except peewee.DoesNotExist:
            return minetorch_pb2.StandardResponse(
                status=1,
                message='Could not find experiment, abort'
            )
        logger = get_runtime_logger(experiment)
        getattr(logger, request.level.lower())(request.log)
        return minetorch_pb2.StandardResponse(
            status=0,
            message='ok'
        )

    def HeyYo(self, request, context):
        print(f'Got a hey from {request.ip_addr}')

        try:
            experiment = Experiment.get_by_id(request.experiment_id)
        except peewee.DoesNotExist:
            return minetorch_pb2.YoMessage(
                roger=True,
                command=C.COMMAND_HALT
            )

        command = C.COMMAND_TRAIN if experiment.is_training == 1 else C.COMMAND_HALT
        return minetorch_pb2.YoMessage(
            roger=True,
            command=command
        )
