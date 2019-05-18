import sys
import time
from concurrent import futures

import grpc

from .grpc import minetorch_pb2_grpc
from .service import MinetorchServicer


class RpcServer():

    def __init__(self, num_worker, address):
        self.num_worker = num_worker
        self.address = address
        self.stop_flag = False

    def serve(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.num_worker))
        minetorch_pb2_grpc.add_MinetorchServicer_to_server(MinetorchServicer(), server)
        server.add_insecure_port(self.address)
        server.start()

        try:
            while True:
                if self.stop_flag:
                    server.top(0)
                    break
                else:
                    time.sleep(3)
        except KeyboardInterrupt:
            server.stop(0)
            sys.exit(0)

    def stop(self):
        self.stop_flag = True
