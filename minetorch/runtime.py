import grpc
from minetorch.rpc.grpc import minetorch_pb2_grpc
from minetorch.rpc.grpc import minetorch_pb2


class RuntimeRpc():

    def __init__(self, addr):
        self.channel = grpc.insecure_channel(addr)
        self.stub = minetorch_pb2_grpc.MinetorchStub(self.channel)

    def heyYo(self):
        message = minetorch_pb2.HeyMessage(
            ip_addr='127.0.0.1',
            status=minetorch_pb2.HeyMessage.Status.Value('IDLE')
        )
        response = self.stub.HeyYo(message)
        print(response)
