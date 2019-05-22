from .grpc import minetorch_pb2_grpc
from .grpc import minetorch_pb2
import logging
from minetorch.orm import Experiment
import minetorch.constants.protobuf_constants as C
import peewee


class MinetorchServicer(minetorch_pb2_grpc.MinetorchServicer):

    def HeyYo(self, request, context):
        logging.info(f'Got a hey from {request.ip_addr}')

        try:
            experiment = Experiment.get_by_id(request.experiment_id)
        except peewee.DoesNotExist:
            return minetorch_pb2.YoMessage(
                roger=True,
                command=C.COMMAND_HALT
            )

        command = C.COMMAND_TRAIN if experiment.is_training == 1 else C.COMMAND_HALT
        print(command, experiment.is_training)
        print(command, experiment.is_training)
        print(command, experiment.is_training)
        print(command, experiment.is_training)
        print(command, experiment.is_training)
        return minetorch_pb2.YoMessage(
            roger=True,
            command=command
        )
