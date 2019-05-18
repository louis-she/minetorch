from .grpc import minetorch_pb2_grpc
from .grpc import minetorch_pb2
import logging


class MinetorchServicer(minetorch_pb2_grpc.MinetorchServicer):

    def HeyYo(self, request, context):
        logging.info(f'Got a hey from {request.ip_addr}')
        return minetorch_pb2.YoMessage(roger=True)
