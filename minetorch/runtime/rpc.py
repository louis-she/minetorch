import grpc
from minetorch.rpc.grpc import minetorch_pb2_grpc
from minetorch.rpc.grpc import minetorch_pb2


class RuntimeRpc():

    def __init__(self, addr):
        self.channel = grpc.insecure_channel(addr)
        self.stub = minetorch_pb2_grpc.MinetorchStub(self.channel)

    def heyYo(self, experiment_id, status):
        message = minetorch_pb2.HeyMessage(
            ip_addr='127.0.0.1',
            status=status,
            experiment_id=experiment_id
        )
        response = self.stub.HeyYo(message)
        return response

    def log(self, experiment_id, record):
        message = minetorch_pb2.Log(
            experiment_id=experiment_id,
            log=record.msg,
            level=record.levelname
        )
        self.stub.SendLog(message)
