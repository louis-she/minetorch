from .grpc import minetorch_pb2_grpc
from .grpc import minetorch_pb2


class MinetorchServicer(minetorch_pb2_grpc.MinetorchServicer):

    def HeyYo(self, request, context):
        return minetorch_pb2.YoMessage(roger=True)
