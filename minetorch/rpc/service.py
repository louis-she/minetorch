from .grpc import minetorch_pb2_grpc
from .grpc import minetorch_pb2
import logging
from minetorch.orm import Experiment
import minetorch.constants.protobuf_constants as C
import peewee


class MinetorchServicer(minetorch_pb2_grpc.MinetorchServicer):

    def SendLog(self, request, context):
        logging.debug(f'Log sent by {request.experiment_id}')
        try:
            experiment = Experiment.get_by_id(request.experiment_id)
        except peewee.DoesNotExist:
            return minetorch_pb2.StandardResponse(
                status=1,
                message='Could not find experiment, abort'
            )
        # TODO: ADD a server log file helper, and write the log to it

    def HeyYo(self, request, context):
        logging.debug(f'Got a hey from {request.ip_addr}')

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
