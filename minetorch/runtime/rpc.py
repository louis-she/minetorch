import grpc
from minetorch.rpc.grpc import minetorch_pb2_grpc
from minetorch.rpc.grpc import minetorch_pb2


def retry(number):
    def decorator(func):
        def __decorator(*args, **kwargs):
            nonlocal number
            for i in range(number - 1):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    continue
            return func(*args, **kwargs)
        return __decorator
    return decorator


class RuntimeRpc():
    """
    TODO: The very first RPC call after forking the child process will fail 100%,
    it will be ok for the second call by retry, not sure why but we can dump the
    network traffic to see what really happens.
    """

    def __init__(self, addr):
        self.channel = grpc.insecure_channel(addr)
        self.stub = minetorch_pb2_grpc.MinetorchStub(self.channel)

    @retry(3)
    def heyYo(self, experiment_id, status):
        message = minetorch_pb2.HeyMessage(
            ip_addr='127.0.0.1',
            status=status,
            experiment_id=experiment_id
        )
        response = self.stub.HeyYo(message)
        return response

    @retry(3)
    def log(self, experiment_id, record):
        message = minetorch_pb2.Log(
            experiment_id=experiment_id,
            log=record.msg,
            level=record.levelname
        )
        self.stub.SendLog(message)
